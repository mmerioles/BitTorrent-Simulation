from __future__ import annotations

import random
from collections import OrderedDict

import networkx as nx
import numpy as np

from .config import SimulationConfig
from .node import Node


class BitTorrentSimulation:
    def __init__(self, config: SimulationConfig | None = None, seed: int | None = None):
        self.config = config or SimulationConfig()
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.graph = nx.Graph()
        self.total_nodes: OrderedDict[int, Node] = OrderedDict()
        self.node_number_tracker = 1
        self.time_counter = 0
        self._build_initial_nodes()
        self.optimistically_unchoke_peers()

    def _build_initial_nodes(self) -> None:
        for _ in range(self.config.initial_number_of_nodes):
            self.total_nodes[self.node_number_tracker] = self._create_node(
                self.node_number_tracker,
                start_with_chunks=True,
            )
            self.graph.add_node(self.node_number_tracker)
            self.node_number_tracker += 1

    def _create_node(self, node_number: int, start_with_chunks: bool) -> Node:
        return Node.create(
            node_number=node_number,
            size_of_file=self.config.size_of_file,
            min_upload_rate=self.config.min_upload_rate,
            max_upload_rate=self.config.max_upload_rate,
            min_initial_chunks=self.config.min_initial_chunks,
            max_initial_chunks=self.config.max_initial_chunks,
            start_with_chunks=start_with_chunks,
        )

    def identify_highest_rate_peers(self, node: Node) -> dict[int, int]:
        highest_rate_peers: dict[int, int] = {}
        for node_number in node.connected_to:
            upload_rate = self.total_nodes[node_number].upload_rate
            if len(highest_rate_peers) < self.config.peer_optimistic_selection_count:
                highest_rate_peers[node_number] = upload_rate
                continue

            lowest_value_node = min(highest_rate_peers, key=highest_rate_peers.get)
            if highest_rate_peers[lowest_value_node] < upload_rate:
                highest_rate_peers.pop(lowest_value_node)
                highest_rate_peers[node_number] = upload_rate

        return highest_rate_peers

    def identify_rarest_chunk_among_best_connections(self, node: Node) -> tuple[int | None, int | None]:
        chunk_frequency: dict[int, int] = {}
        peer_chunk_source: dict[int, int] = {}
        own_chunks = set(node.chunks)

        for node_number in self.identify_highest_rate_peers(node):
            connected_node_chunks = self.total_nodes[node_number].chunks
            for chunk in connected_node_chunks:
                if chunk not in own_chunks:
                    chunk_frequency[chunk] = chunk_frequency.get(chunk, 0) + 1
                    peer_chunk_source[chunk] = node_number

        if not chunk_frequency:
            return (None, None)

        rarest_chunk = min(chunk_frequency, key=chunk_frequency.get)
        return (rarest_chunk, peer_chunk_source[rarest_chunk])

    def exchange_data_for_node(self, node: Node) -> None:
        for _ in range(node.upload_rate):
            rarest_chunk, provider_node = self.identify_rarest_chunk_among_best_connections(node)
            if rarest_chunk is None or len(node.chunks) >= self.config.size_of_file:
                continue

            if rarest_chunk not in node.chunks:
                node.chunks.append(rarest_chunk)

            if not node.chunks:
                continue

            chunk_to_send = random.choice(node.chunks)
            provider_chunks = self.total_nodes[provider_node].chunks
            if chunk_to_send not in provider_chunks:
                provider_chunks.append(chunk_to_send)

    def optimistically_unchoke_peers(self) -> None:
        for node in self.total_nodes.values():
            node.connected_to.clear()

        self.graph.clear_edges()

        for node_number, node in self.total_nodes.items():
            while len(node.connected_to) < self.config.total_number_of_connections:
                possible_connections = [
                    num
                    for num in self.total_nodes
                    if num != node_number and num not in node.connected_to
                ]
                if not possible_connections:
                    break

                connect_to = random.choice(possible_connections)
                node.add_connection(connect_to)
                self.total_nodes[connect_to].add_connection(node_number)

            for connect_to in node.connected_to:
                self.graph.add_edge(node_number, connect_to)

    def update(self) -> None:
        self.time_counter += 1

        if self.time_counter % self.config.optimistically_unchoke_constant == 0:
            self.optimistically_unchoke_peers()

        for node in self.total_nodes.values():
            self.exchange_data_for_node(node)

    def add_node(self) -> int:
        node_id = self.node_number_tracker
        new_node = self._create_node(node_id, start_with_chunks=False)
        self.total_nodes[node_id] = new_node
        self.graph.add_node(node_id)

        possible_connections = list(self.total_nodes.keys())[:-1]
        connections = random.sample(
            possible_connections,
            min(self.config.total_number_of_connections, len(possible_connections)),
        )
        for connect_to in connections:
            new_node.add_connection(connect_to)
            self.total_nodes[connect_to].add_connection(node_id)
            self.graph.add_edge(node_id, connect_to)

        self.node_number_tracker += 1
        self.update()
        return node_id

    def delete_node(self) -> int | None:
        if not self.graph.nodes:
            return None

        node_to_remove = random.choice(list(self.graph.nodes()))
        self.total_nodes.pop(node_to_remove, None)
        for node in self.total_nodes.values():
            node.remove_connection(node_to_remove)

        self.graph.remove_node(node_to_remove)
        self.update()
        return node_to_remove

    def completion_ratio(self, node: Node) -> float:
        return len(node.chunks) / self.config.size_of_file

    def completed_nodes(self) -> int:
        return sum(1 for node in self.total_nodes.values() if len(node.chunks) == self.config.size_of_file)

    def summary(self) -> dict[str, int]:
        return {
            "time_counter": self.time_counter,
            "total_nodes": len(self.total_nodes),
            "completed_nodes": self.completed_nodes(),
            "graph_edges": self.graph.number_of_edges(),
        }

    def node_rows(self) -> list[str]:
        rows: list[str] = []
        for node in self.total_nodes.values():
            connected_to_formatted = ", ".join(f"{num:2d}" for num in sorted(node.connected_to))
            chunks_formatted = ", ".join(f"{chunk:2d}" for chunk in sorted(node.chunks))
            rarest_chunk, _ = self.identify_rarest_chunk_among_best_connections(node)
            rarest_chunk_formatted = f"{rarest_chunk:2d}" if rarest_chunk is not None else "None"
            rows.append(
                f"Node: {node.node_number:2d} | Connected to: [{connected_to_formatted:<50}] | "
                f"Chunks: [{chunks_formatted:<40}] | RAREST CHUNK: {rarest_chunk_formatted} | "
                f"DOWNLOAD COMPLETION PERCENTAGE: {round(self.completion_ratio(node) * 100, 2)}% | "
                f"UPLOAD RATE: {node.upload_rate} | BEST PEERS: {self.identify_highest_rate_peers(node)}"
            )
        return rows
