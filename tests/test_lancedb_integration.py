import unittest
import unittest.mock
import os
import shutil
import sys
import time
# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from lazzaro.core.memory_system import MemorySystem
from lazzaro.core.memory_shard import MemoryShard
from lazzaro.models.graph import Node

class TestLanceDBIntegration(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_int_db"
        if os.path.exists(self.test_db):
            shutil.rmtree(self.test_db)
        # We need a real embedder or a mocked one. MemorySystem defaults to OpenAI.
        # Let's mock the embedder to avoid API calls.
        mock_embedder = unittest.mock.MagicMock()
        mock_embedder.embed.return_value = [0.1] * 1536
        mock_embedder.batch_embed.return_value = [[0.1] * 1536]
        
        self.ms = MemorySystem(
            openai_api_key="fake",
            db_dir=self.test_db,
            embedding_provider=mock_embedder,
            load_from_disk=False
        )

    def tearDown(self):
        if hasattr(self, 'ms'):
            self.ms.vector_store.close()
        if os.path.exists(self.test_db):
            shutil.rmtree(self.test_db)

    def test_sync_on_load(self):
        # 1. Create a node manually in the graph but NOT in LanceDB
        node_id = "manual_node"
        node = Node(
            id=node_id,
            content="Manual content",
            embedding=[0.2] * 1536,
            shard_key="default"
        )
        shard = MemoryShard("default")
        shard.add_node(node)
        self.ms.shards["default"] = shard
        self.ms.buffer.nodes[node_id] = node
        
        # 2. Save to persistence (Pickle)
        self.ms._save_to_persistence()
        
        # 3. Shutdown and restart
        del self.ms
        ms2 = MemorySystem(
            openai_api_key="fake",
            db_dir=self.test_db,
            load_from_disk=True
        )
        
        # 4. Verify node is in MS2 (from Pickle)
        self.assertIn(node_id, ms2.buffer.nodes)
        
        # 5. Verify node is NOW in LanceDB (via search)
        results = ms2.vector_store.search([0.2] * 1536, limit=1)
        self.assertIn(node_id, results)
        ms2.vector_store.close()

    def test_delete_sync(self):
        # 1. Add node through normal process
        self.ms.start_conversation()
        self.ms.add_to_short_term("Delete me later")
        # Mock consolidation to add node
        node_id = "node_to_delete"
        node = Node(id=node_id, content="Delete me later", embedding=[0.3]*1536)
        self.ms._get_or_create_shard("default").add_node(node)
        self.ms.vector_store.add([{
            "id": node_id, "content": node.content, "embedding": node.embedding
        }])
        
        # 2. Verify it's there
        self.assertIn(node_id, self.ms.vector_store.search([0.3]*1536))
        
        # 3. Trigger buffer limit enforcement (force deletion)
        self.ms.max_buffer_size = 0 
        self.ms._enforce_buffer_limit()
        
        # 4. Verify it's gone from LanceDB
        self.assertNotIn(node_id, self.ms.vector_store.search([0.3]*1536))

if __name__ == '__main__':
    unittest.main()
