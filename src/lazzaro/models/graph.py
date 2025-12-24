from dataclasses import dataclass, field, asdict
from typing import List, Optional
import time

@dataclass
class Node:
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

    def to_dict(self):
        return asdict(self)


@dataclass
class Edge:
    source: str
    target: str
    weight: float = 1.0
    edge_type: str = "relates_to"
    co_occurrence: int = 1
    last_updated: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)
