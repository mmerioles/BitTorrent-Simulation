from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Node:
    node_number: int
    chunks: list[int]
    upload_rate: int
    connected_to: list[int] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        node_number: int,
        size_of_file: int,
        min_upload_rate: int,
        max_upload_rate: int,
        min_initial_chunks: int,
        max_initial_chunks: int,
        start_with_chunks: bool = True,
    ) -> "Node":
        chunks: list[int] = []
        if start_with_chunks:
            num_chunks = random.randint(min_initial_chunks, max_initial_chunks)
            chunks = random.sample(range(1, size_of_file + 1), num_chunks)

        return cls(
            node_number=node_number,
            chunks=chunks,
            upload_rate=random.randint(min_upload_rate, max_upload_rate),
        )

    def add_connection(self, other_node_number: int) -> None:
        if other_node_number not in self.connected_to:
            self.connected_to.append(other_node_number)

    def remove_connection(self, other_node_number: int) -> None:
        if other_node_number in self.connected_to:
            self.connected_to.remove(other_node_number)
