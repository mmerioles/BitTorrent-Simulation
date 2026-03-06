from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .simulation import BitTorrentSimulation


def build_layout(
    simulation: BitTorrentSimulation,
    initial_pos: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    node_count = simulation.graph.number_of_nodes()
    if node_count == 0:
        return {}

    spring_k = 2.4 / np.sqrt(node_count)
    return nx.spring_layout(
        simulation.graph,
        pos=initial_pos,
        seed=7,
        k=spring_k,
        iterations=100,
        scale=3.0,
    )


def ensure_layout(
    simulation: BitTorrentSimulation,
    pos: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    if pos is None:
        return build_layout(simulation)

    updated_pos = {node: value for node, value in pos.items() if node in simulation.graph.nodes}
    return build_layout(simulation, initial_pos=updated_pos or None)


def draw_simulation(
    simulation: BitTorrentSimulation,
    pos: dict[int, np.ndarray] | None = None,
    title: str = "BitTorrent Simulation",
):
    pos = ensure_layout(simulation, pos)
    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_on_axis(simulation, ax=ax, pos=pos, title=title)
    fig.tight_layout()
    return fig, ax, pos


def _draw_on_axis(
    simulation: BitTorrentSimulation,
    ax,
    pos: dict[int, np.ndarray],
    title: str,
) -> None:
    node_colors = [
        "green"
        if len(simulation.total_nodes[node].chunks) == simulation.config.size_of_file
        else "skyblue"
        for node in simulation.graph.nodes()
    ]

    nx.draw_networkx_nodes(simulation.graph, pos, ax=ax, node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(simulation.graph, pos, ax=ax)
    nx.draw_networkx_labels(simulation.graph, pos, font_size=10, ax=ax)
    ax.set_title(title)
    ax.text(-0.05, 1.02, f"Time Count: {simulation.time_counter}", transform=ax.transAxes, ha="left")
    ax.axis("off")


def refresh_axis(
    simulation: BitTorrentSimulation,
    ax,
    pos: dict[int, np.ndarray] | None = None,
    title: str = "BitTorrent Simulation",
) -> dict[int, np.ndarray]:
    pos = ensure_layout(simulation, pos)
    ax.clear()
    _draw_on_axis(simulation, ax=ax, pos=pos, title=title)
    return pos


def save_snapshot(
    simulation: BitTorrentSimulation,
    output_path: str | Path,
    pos: dict[int, np.ndarray] | None = None,
    title: str = "BitTorrent Simulation",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _, pos = draw_simulation(simulation, pos=pos, title=title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
