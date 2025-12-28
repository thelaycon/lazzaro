import time
from typing import Dict, List, Optional, Tuple, Set
from ..models.graph import Node, Edge
from .memory_shard import MemoryShard

class BufferGraph:
    """
    The composite graph view that spans across all memory shards and super-nodes.

    BufferGraph provides a unified interface for graph operations (DFS, pruning,
    neighbor retrieval) regardless of which semantic shard the data resides in.

    Args:
        shards: Dictionary mapping shard keys to MemoryShard instances.
        super_nodes: Dictionary mapping node IDs to their super-node representations.

    Example:
        ```python
        bg = BufferGraph(shards={"work": shard1}, super_nodes={})
        summary = bg.get_all_nodes_summary()
        ```
    """

    def __init__(self, shards: Dict[str, MemoryShard], super_nodes: Dict[str, Node]):
        self.shards = shards
        self.super_nodes = super_nodes

    @property
    def nodes(self) -> Dict[str, Node]:
        """Returns a combined dictionary of all nodes across all shards and super-nodes."""
        all_nodes = dict(self.super_nodes)
        for shard in self.shards.values():
            all_nodes.update(shard.nodes)
        return all_nodes

    @property
    def edges(self) -> Dict[Tuple[str, str], Edge]:
        """Returns a combined dictionary of all edges across all shards."""
        all_edges = {}
        for shard in self.shards.values():
            all_edges.update(shard.edges)
        return all_edges

    def add_node(self, node: Node):
        """Dispatches a node to its appropriate semantic shard."""
        shard_key = node.shard_key or "default"
        if shard_key not in self.shards:
            self.shards[shard_key] = MemoryShard(shard_key)
        self.shards[shard_key].add_node(node)

    def add_edge(self, edge: Edge):
        """
        Dispatches an edge to the shard containing its source node.
        If the source is not localized to a shard, it defaults to the 'default' shard.
        """
        for shard in self.shards.values():
            if edge.source in shard.nodes:
                shard.add_edge(edge)
                return
        if "default" in self.shards:
            self.shards["default"].add_edge(edge)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieves a node by ID from any shard or the super-node registry."""
        if node_id in self.super_nodes:
            return self.super_nodes[node_id]
        for shard in self.shards.values():
            if node_id in shard.nodes:
                return shard.nodes[node_id]
        return None

    def get_neighbors(self, node_id: str, min_weight: float = 0.3) -> List[str]:
        """Retrieves neighbors for a node from the shard it resides in."""
        for shard in self.shards.values():
            if node_id in shard.nodes:
                return shard.get_neighbors(node_id, min_weight)
        return []

    def update_access(self, node_id: str):
        """Updates frequency metrics and salience for a node upon successful retrieval."""
        node = self.get_node(node_id)
        if node:
            node.access_count += 1
            node.last_accessed = time.time()
            node.salience = min(1.0, node.salience + 0.05)

    def apply_temporal_decay(self, decay_rate: float = 0.01):
        """Signals all shards to apply temporal decay to their respective subgraphs."""
        for shard in self.shards.values():
            shard.apply_temporal_decay(decay_rate)

    def prune_weak_edges(self, threshold: float = 0.5) -> int:
        """Triggers edge pruning across all shards and returns total pruned count."""
        total = 0
        for shard in self.shards.values():
            total += shard.prune_weak_edges(threshold)
        return total

    def get_connected_components(self) -> List[Set[str]]:
        """
        Performs a Depth-First Search (DFS) to find all isolated memory clusters.
        Useful for identifying related memories for profile extraction.
        """
        visited = set()
        components = []

        def dfs(node_id: str, component: Set[str]):
            visited.add(node_id)
            component.add(node_id)
            for neighbor in self.get_neighbors(node_id, min_weight=0.0):
                if neighbor not in visited:
                    dfs(neighbor, component)

        for node_id in self.nodes:
            if node_id not in visited:
                component = set()
                dfs(node_id, component)
                components.append(component)

        return components

    def size(self) -> Tuple[int, int]:
        """Returns the total number of (nodes, edges) in the entire memory system."""
        total_nodes = sum(len(s.nodes) for s in self.shards.values()) + len(self.super_nodes)
        total_edges = sum(len(s.edges) for s in self.shards.values())
        return total_nodes, total_edges

    def get_all_nodes_summary(self) -> List[Dict]:
        """Returns a truncated summary of all memories for display or debugging purposes."""
        all_nodes = list(self.nodes.values())
        return [
            {
                "id": node.id,
                "content": node.content[:100] + "..." if len(node.content) > 100 else node.content,
                "type": node.type,
                "salience": node.salience,
                "access_count": node.access_count,
                "shard": node.shard_key
            }
            for node in sorted(all_nodes, key=lambda n: n.timestamp, reverse=True)
        ]
