import os
import lancedb
import pyarrow as pa
from typing import List, Dict, Any, Optional

class LanceDBVectorStore:
    """
    LanceDB implementation of the VectorStore interface.
    
    This provides efficient vector search for memory nodes using LanceDB.
    """
    def __init__(self, db_dir: str = "db", table_name: str = "memories"):
        self.db_dir = db_dir
        self.table_name = table_name
        self._uri = os.path.join(db_dir, "lancedb")
        os.makedirs(self._uri, exist_ok=True)
        self.db = lancedb.connect(self._uri)
        self._table = None
        self._ensure_table()

    def _ensure_table(self):
        """Ensures the table exists with the correct schema."""
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 1536)), # Defaulting to OpenAI size
            pa.field("type", pa.string()),
            pa.field("salience", pa.float32()),
            pa.field("shard_key", pa.string()),
            pa.field("timestamp", pa.float64())
        ])
        
        if self.table_name in self.db.table_names():
            self._table = self.db.open_table(self.table_name)
        else:
            # We initialize with an empty list and schema if table doesn't exist
            self._table = self.db.create_table(self.table_name, schema=schema)

    def add(self, nodes: List[Dict[str, Any]]):
        """Add a batch of nodes to LanceDB."""
        if not nodes:
            return
        
        # Convert nodes to the format expected by the schema
        data = []
        for node in nodes:
            data.append({
                "id": node["id"],
                "content": node["content"],
                "vector": node["embedding"],
                "type": node.get("type", "semantic"),
                "salience": float(node.get("salience", 0.5)),
                "shard_key": node.get("shard_key", "default"),
                "timestamp": node.get("timestamp", 0.0)
            })
        
        self._table.add(data)

    def search(self, query_emb: List[float], limit: int = 5) -> List[str]:
        """Search for the most similar nodes."""
        if self._table is None:
            return []
        
        results = self._table.search(query_emb).limit(limit).to_list()
        return [r["id"] for r in results]

    def delete(self, node_ids: List[str]):
        """Delete specific nodes by their IDs."""
        if not node_ids:
            return
        # LanceDB uses SQL-like filters for deletion
        filter_str = "id IN (" + ",".join([f"'{node_id}'" for node_id in node_ids]) + ")"
        self._table.delete(filter_str)

    def close(self):
        """LanceDB handles connections automatically, but we can provide this for the interface."""
        pass
