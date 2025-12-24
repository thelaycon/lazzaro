import time
from typing import Dict, Tuple, List
from ..models.graph import Node, Edge

class MemoryShard:
    def __init__(self, shard_key: str):
        self.shard_key = shard_key
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str], Edge] = {}
        self.last_accessed = time.time()
        self.access_count = 0

    def add_node(self, node: Node):
        node.shard_key = self.shard_key
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        key = (edge.source, edge.target)
        if key in self.edges:
            self.edges[key].weight = min(1.0, self.edges[key].weight + 0.1)
            self.edges[key].co_occurrence += 1
        else:
            self.edges[key] = edge

    def get_neighbors(self, node_id: str, min_weight: float = 0.3) -> List[str]:
        neighbors = []
        for (src, tgt), edge in self.edges.items():
            if src == node_id and edge.weight >= min_weight:
                neighbors.append(tgt)
            elif tgt == node_id and edge.weight >= min_weight:
                neighbors.append(src)
        return neighbors

    def apply_temporal_decay(self, decay_rate: float = 0.01) -> None:
        """Applies Non-Linear Sigmoidal Decay"""
        for edge in self.edges.values():
            edge.weight *= (1 - decay_rate)

        for node in self.nodes.values():
            # NEW: Sigmoidal/Asymptotic Decay flattening at 0.2
            # Instead of simple multiplication, we decay the portion above 0.2
            # This ensures node.salience never mathematically drops below 0.2
            floor = 0.2
            if node.salience > floor:
                # The distance to the floor decays exponentially
                node.salience = floor + (node.salience - floor) * (1 - decay_rate)
            else:
                node.salience = floor

    def prune_weak_edges(self, threshold: float = 0.5) -> int:
        to_remove = [k for k, e in self.edges.items() if e.weight < threshold]
        for key in to_remove:
            del self.edges[key]
        return len(to_remove)

    def size(self) -> Tuple[int, int]:
        return len(self.nodes), len(self.edges)
