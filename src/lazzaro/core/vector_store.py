import os
import lancedb
import pyarrow as pa
from typing import List, Dict, Any, Optional
import json

class LanceDBStore:
    """
    LanceDB implementation of the Store protocol for v0.3.0.
    
    This provides persistent storage for nodes, edges, and profiles,
    replacing both the legacy VectorStore and Pickle persistence.
    """
    def __init__(self, db_dir: str = "db"):
        self.db_dir = db_dir
        self._uri = os.path.join(db_dir, "lancedb")
        os.makedirs(self._uri, exist_ok=True)
        self.db = lancedb.connect(self._uri)
        
        self.nodes_table_name = "nodes"
        self.edges_table_name = "edges"
        self.profile_table_name = "profiles"
        
        self._nodes_table = None
        self._edges_table = None
        self._profile_table = None
        
        self._ensure_tables()

    def _ensure_tables(self):
        """Ensures all tables exist with correct schemas."""
        # 1. Nodes Table
        node_schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 1536)),
            pa.field("type", pa.string()),
            pa.field("timestamp", pa.float64()),
            pa.field("access_count", pa.int32()),
            pa.field("last_accessed", pa.float64()),
            pa.field("salience", pa.float32()),
            pa.field("is_super_node", pa.bool_()),
            pa.field("child_ids", pa.string()), # JSON list
            pa.field("parent_id", pa.string()),
            pa.field("shard_key", pa.string()),
            pa.field("metadata", pa.string()) # Stored as JSON string
        ])
        
        if self.nodes_table_name in self.db.list_tables():
            self._nodes_table = self.db.open_table(self.nodes_table_name)
        else:
            self._nodes_table = self.db.create_table(self.nodes_table_name, schema=node_schema, exist_ok=True)
            # Create scalar index for fast multi-tenant filtering
            self._nodes_table.create_scalar_index("user_id", index_type="BTREE")

        # 2. Edges Table
        edge_schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("user_id", pa.string()),
            pa.field("source_id", pa.string()),
            pa.field("target_id", pa.string()),
            pa.field("weight", pa.float32()),
            pa.field("edge_type", pa.string()),
            pa.field("co_occurrence", pa.int32()),
            pa.field("last_updated", pa.float64()),
            pa.field("metadata", pa.string())
        ])
        
        if self.edges_table_name in self.db.list_tables():
            self._edges_table = self.db.open_table(self.edges_table_name)
        else:
            self._edges_table = self.db.create_table(self.edges_table_name, schema=edge_schema, exist_ok=True)

        # 3. Profile Table
        profile_schema = pa.schema([
            pa.field("user_id", pa.string()),
            pa.field("data", pa.string()), # JSON blob
            pa.field("updated_at", pa.float64())
        ])
        
        if self.profile_table_name in self.db.list_tables():
            self._profile_table = self.db.open_table(self.profile_table_name)
        else:
            self._profile_table = self.db.create_table(self.profile_table_name, schema=profile_schema, exist_ok=True)

    # --- Node Operations ---

    def add_nodes(self, nodes: List[Dict[str, Any]], user_id: str = "default"):
        if not nodes:
            return
        
        data = []
        for node in nodes:
            data.append({
                "id": node["id"],
                "user_id": user_id,
                "content": node["content"],
                "vector": node["embedding"],
                "type": node.get("type", "semantic"),
                "timestamp": node.get("timestamp", 0.0),
                "access_count": node.get("access_count", 0),
                "last_accessed": node.get("last_accessed", 0.0),
                "salience": float(node.get("salience", 0.5)),
                "is_super_node": node.get("is_super_node", False),
                "child_ids": json.dumps(node.get("child_ids", [])),
                "parent_id": node.get("parent_id") or "",
                "shard_key": node.get("shard_key", "default"),
                "metadata": json.dumps(node.get("metadata", {}))
            })
        
        self._nodes_table.add(data)

    def get_nodes(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """Retrieve all nodes for a specific user."""
        if self._nodes_table is None:
            return []
        results = self._nodes_table.search().where(f"user_id = '{user_id}'").to_list()
        for r in results:
            if "child_ids" in r and isinstance(r["child_ids"], str):
                try:
                    r["child_ids"] = json.loads(r["child_ids"])
                except:
                    r["child_ids"] = []
            if "metadata" in r and isinstance(r["metadata"], str):
                try:
                    r["metadata"] = json.loads(r["metadata"])
                except:
                    r["metadata"] = {}
        return results

    def search_nodes(self, query_emb: List[float], user_id: str = "default", limit: int = 5) -> List[str]:
        if self._nodes_table is None:
            return []
        
        results = self._nodes_table.search(query_emb)\
            .where(f"user_id = '{user_id}'")\
            .limit(limit)\
            .to_list()
        return [r["id"] for r in results]

    def delete_nodes(self, node_ids: List[str], user_id: str = "default"):
        if node_ids is None or len(node_ids) == 0:
            # Clear all nodes for this user
            filter_str = f"user_id = '{user_id}'"
        else:
            filter_str = f"user_id = '{user_id}' AND id IN (" + ",".join([f"'{nid}'" for nid in node_ids]) + ")"
        self._nodes_table.delete(filter_str)

    def get_latest_version(self) -> int:
        """Returns the latest version number of the nodes table."""
        if self._nodes_table is None:
            return 0
        # Re-opening/Refresh to ensure we see cross-process updates
        self._nodes_table = self.db.open_table(self.nodes_table_name)
        return self._nodes_table.version

    # --- Edge Operations ---
    def add_edges(self, edges: List[Dict[str, Any]], user_id: str = "default"):
        if not edges:
            return
        
        data = []
        for edge in edges:
            src = edge.get("source") or edge.get("source_id")
            tgt = edge.get("target") or edge.get("target_id")
            e_type = edge.get("edge_type") or edge.get("type", "relates_to")
            
            data.append({
                "id": edge.get("id", f"{src}_{tgt}"),
                "user_id": user_id,
                "source_id": src,
                "target_id": tgt,
                "weight": float(edge.get("weight", 1.0)),
                "edge_type": e_type,
                "co_occurrence": edge.get("co_occurrence", 1),
                "last_updated": edge.get("last_updated", 0.0),
                "metadata": json.dumps(edge.get("metadata", {}))
            })
        
        self._edges_table.add(data)

    def delete_edges(self, source_id: Optional[str] = None, user_id: str = "default"):
        if self._edges_table is None:
            return
        
        filter_str = f"user_id = '{user_id}'"
        if source_id:
            filter_str += f" AND source_id = '{source_id}'"
            
        self._edges_table.delete(filter_str)

    def get_edges(self, source_id: Optional[str] = None, user_id: str = "default") -> List[Dict[str, Any]]:
        if self._edges_table is None:
            return []
        
        query = self._edges_table.search().where(f"user_id = '{user_id}'")
        if source_id:
            query = query.where(f"source_id = '{source_id}'")
            
        results = query.to_list()
        for r in results:
            # Map edge_type to type for backward compatibility in the dict
            if "edge_type" in r:
                r["type"] = r["edge_type"]
            
            if "metadata" in r and isinstance(r["metadata"], str):
                try:
                    r["metadata"] = json.loads(r["metadata"])
                except:
                    r["metadata"] = {}
        return results

    # --- Profile Operations ---

    def save_profile(self, profile_data: Dict[str, Any], user_id: str = "default"):
        import time
        data = [{
            "user_id": user_id,
            "data": json.dumps(profile_data),
            "updated_at": time.time()
        }]
        
        # First delete existing profile for this user
        self._profile_table.delete(f"user_id = '{user_id}'")
        # Then add new
        self._profile_table.add(data)

    def load_profile(self, user_id: str = "default") -> Optional[Dict[str, Any]]:
        results = self._profile_table.search().where(f"user_id = '{user_id}'").to_list()
        if results:
            return json.loads(results[0]["data"])
        return None

    def close(self):
        """Close the LanceDB connection."""
        pass 
        # LanceDB doesn't have a formal close() on the DB object in all versions, 
        # but the table objects are handles.
        # We'll just null them out to help GC.
        self._nodes_table = None
        self._edges_table = None
        self._profile_table = None
        self.db = None
