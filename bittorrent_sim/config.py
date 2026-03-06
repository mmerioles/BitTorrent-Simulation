from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    size_of_file: int = 30
    initial_number_of_nodes: int = 100
    peer_optimistic_selection_count: int = 2
    total_number_of_connections: int = 4
    optimistically_unchoke_constant: int = 4
    min_initial_chunks: int = 2
    max_initial_chunks: int = 3
    min_upload_rate: int = 1
    max_upload_rate: int = 5
