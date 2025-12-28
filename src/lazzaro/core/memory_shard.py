import time
from typing import Dict, List, Tuple

from ..models.graph import Edge, Node


class MemoryShard:
    """
    Manages a semantically isolated subgraph of memories.

    Memory Sharding allows the system to scale by organizing memories into
    thematic clusters (e.g., "work", "health"). This reduces search space
    and improves retrieval precision.

    Attributes:
        shard_key: The semantic identifier for this shard.
        nodes: Mapping of node IDs to Node objects in this shard.
        edges: Mapping of (source, target) tuples to Edge objects.
        last_accessed: Unix timestamp of the last retrieval from this shard.
        access_count: Total number of retrievals performed on this shard.

    Example:
        ```python
        shard = MemoryShard("programming")
        shard.add_node(node)
        shard.add_edge(edge)
        ```
    """

    def __init__(self, shard_key: str):
        self.shard_key = shard_key
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str], Edge] = {}
        self.last_accessed = time.time()
        self.access_count = 0

    def add_node(self, node: Node) -> None:
        """Adds a node to the shard, inheriting the shard's semantic key."""
        node.shard_key = self.shard_key
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """
        Adds or strengthens an association between two nodes in the shard.
        If the edge already exists, its weight and co-occurrence count are increased.
        """
        key = (edge.source, edge.target)
        if key in self.edges:
            self.edges[key].weight = min(1.0, self.edges[key].weight + 0.1)
            self.edges[key].co_occurrence += 1
        else:
            self.edges[key] = edge

    def get_neighbors(self, node_id: str, min_weight: float = 0.3) -> List[str]:
        """Returns IDs of nodes connected to the given node with sufficient weight."""
        neighbors = []
        for (src, tgt), edge in self.edges.items():
            if src == node_id and edge.weight >= min_weight:
                neighbors.append(tgt)
            elif tgt == node_id and edge.weight >= min_weight:
                neighbors.append(src)
        return neighbors

    def apply_temporal_decay(self, decay_rate: float = 0.01) -> None:
        """
        Applies non-linear decay to node salience and edge weights.
        Node salience follows an asymptotic decay curve flattening at 0.2.
        """
        for edge in self.edges.values():
            edge.weight *= (1 - decay_rate)

        for node in self.nodes.values():
            floor = 0.2
            if node.salience > floor:
                node.salience = floor + (node.salience - floor) * (1 - decay_rate)
            else:
                node.salience = floor

    def prune_weak_edges(self, threshold: float = 0.5) -> int:
        """Removes all associative edges with weight below the threshold."""
        to_remove = [k for k, e in self.edges.items() if e.weight < threshold]
        for key in to_remove:
            del self.edges[key]
        return len(to_remove)

    def size(self) -> Tuple[int, int]:
        """Returns the number of (nodes, edges) in this shard."""
        return len(self.nodes), len(self.edges)
