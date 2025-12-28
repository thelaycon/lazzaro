import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class Node:
    """
    Represents an atomic memory unit (Node N) within the memory graph.

    A Node stores the actual content of a memory, its vector embedding for semantic
    retrieval, and various biological-inspired metrics like salience and access frequency.

    Attributes:
        id: Unique identifier for the node.
        content: The text content of the memory.
        embedding: Vector representation for similarity searches.
        type: Category of memory (e.g., "semantic", "episodic", "procedural").
        timestamp: Creation time of the memory node.
        access_count: Number of times this memory has been retrieved.
        last_accessed: Unix timestamp of the last retrieval.
        salience: Importance score [0.0 - 1.0], decays over time.
        is_super_node: Whether this node represents a cluster of other nodes.
        child_ids: List of node IDs if this is a super-node.
        parent_id: ID of the super-node containing this node, if any.
        shard_key: The semantic category/shard this node belongs to.

    Example:
        ```python
        node = Node(
            id="mem_123",
            content="User prefers Python over Java",
            embedding=[0.1, 0.2, ...],
            shard_key="programming"
        )
        ```
    """

    id: str
    content: str
    embedding: List[float] = field(default_factory=list)
    type: str = "semantic"
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    salience: float = 0.5
    is_super_node: bool = False
    child_ids: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    shard_key: str = "default"

    @classmethod
    def from_dict(cls, data: dict):
        """Creates a Node from a dictionary."""
        # Use only existing fields in data that match Node's __init__
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        """Returns a serializable dictionary of the node."""
        return asdict(self)


@dataclass
class Edge:
    """
    Represents a directed associative connection (Edge E) between two Nodes.

    Edges track the strength of relationships between memories based on
    semantic similarity and temporal co-occurrence.

    Attributes:
        source: The ID of the source Node.
        target: The ID of the target Node.
        weight: The strength of the association [0.0 - 1.0].
        edge_type: The nature of the relationship (e.g., "relates_to").
        co_occurrence: Frequency of these nodes appearing in the same context.
        last_updated: Unix timestamp of the last weight update.

    Example:
        ```python
        edge = Edge(
            source="mem_python",
            target="mem_rust",
            weight=0.8,
            edge_type="alternative_to"
        )
        ```
    """

    source: str
    target: str
    weight: float = 1.0
    edge_type: str = "relates_to"
    co_occurrence: int = 1
    last_updated: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, data: dict):
        """Creates an Edge from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        """Returns a serializable dictionary of the edge."""
        return asdict(self)
