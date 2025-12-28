import unittest
import os
import shutil
import time
from unittest.mock import MagicMock
from lazzaro.core.memory_system import MemorySystem
from lazzaro.core.memory_shard import MemoryShard
from lazzaro.models.graph import Node, Edge

class MockLLM:
    def completion(self, messages, response_format=None): return "{}"
    def completion_stream(self, messages, response_format=None): yield ""

class MockEmbedder:
    def embed(self, text): return [0.1]*1536
    def batch_embed(self, texts): return [[0.1]*1536 for _ in texts]

class TestV03Migration(unittest.TestCase):
    def setUp(self):
        self.db_dir = "test_v03_db"
        if os.path.exists(self.db_dir):
            shutil.rmtree(self.db_dir)
        os.makedirs(self.db_dir)
        self.mock_llm = MockLLM()
        self.mock_embedder = MockEmbedder()

    def tearDown(self):
        if hasattr(self, "ms_a"): self.ms_a.close()
        if hasattr(self, "ms_b"): self.ms_b.close()
        time.sleep(0.5) # Give it a moment to release handles
        if os.path.exists(self.db_dir):
            try:
                shutil.rmtree(self.db_dir)
            except:
                print(f"DEBUG: Failed to remove {self.db_dir}")

    def test_full_persistence_and_sync(self):
        """Test that nodes, edges, and profile are correctly persisted and reloaded in v0.3.0."""
        # 1. Setup Instance A and add data
        self.ms_a = MemorySystem(
            db_dir=self.db_dir, 
            user_id="test_user",
            llm_provider=self.mock_llm,
            embedding_provider=self.mock_embedder
        )
        
        # Add a node manually to skip LLM calls for speed
        node1 = Node(id="node_1", content="Fact 1", embedding=[0.1]*1536, shard_key="work")
        self.ms_a.shards["work"] = MemoryShard("work")
        self.ms_a.shards["work"].add_node(node1)
        
        edge1 = Edge(source="node_1", target="node_X", weight=0.9) # node_X doesn't exist but let's test persistence
        self.ms_a.shards["work"].add_edge(edge1)
        
        self.ms_a.profile.update_domain("preferences", "Loves minimalism")
        
        # Trigger save
        self.ms_a._save_to_persistence()
        
        # 2. Setup Instance B and load data
        self.ms_b = MemorySystem(
            db_dir=self.db_dir, 
            user_id="test_user", 
            load_from_disk=True,
            llm_provider=self.mock_llm,
            embedding_provider=self.mock_embedder
        )
        
        self.assertEqual(len(self.ms_b.buffer.nodes), 1)
        self.assertIn("node_1", self.ms_b.buffer.nodes)
        self.assertEqual(self.ms_b.buffer.nodes["node_1"].content, "Fact 1")
        
        # Check edges
        found_edge = False
        for shard in self.ms_b.shards.values():
            if ("node_1", "node_X") in shard.edges:
                found_edge = True
                self.assertAlmostEqual(shard.edges[("node_1", "node_X")].weight, 0.9, places=5)
        self.assertTrue(found_edge)
        
        # Check profile
        self.assertIn("Loves minimalism", self.ms_b.profile.get_context())
        
        # 3. Test multi-process sync (check_for_updates)
        v_before = self.ms_b.store.get_latest_version()
        print(f"DEBUG: ms_b version before ms_a save: {v_before}")
        
        # Instance A adds another node
        node2 = Node(id="node_2", content="Fact 2", embedding=[0.2]*1536, shard_key="personal")
        self.ms_a.shards["personal"] = MemoryShard("personal")
        self.ms_a.shards["personal"].add_node(node2)
        self.ms_a._save_to_persistence()
        
        v_after_a = self.ms_a.store.get_latest_version()
        print(f"DEBUG: ms_a version after save: {v_after_a}")
        
        # Wait a bit for filesystem/DB to settle (though LanceDB is immediate)
        time.sleep(0.5)
        
        v_b_check = self.ms_b.store.get_latest_version()
        print(f"DEBUG: ms_b version seeing: {v_b_check}")
        print(f"DEBUG: ms_b cached _last_nodes_version: {getattr(self.ms_b, '_last_nodes_version', 'N/A')}")
        
        # Instance B should detect change
        updated = self.ms_b.check_for_updates()
        self.assertTrue(updated, f"ms_b did not detect update. version: {v_b_check}, last: {self.ms_b._last_nodes_version}")
        self.assertIn("node_2", self.ms_b.buffer.nodes)
        self.assertEqual(len(self.ms_b.buffer.nodes), 2)

if __name__ == "__main__":
    unittest.main()
