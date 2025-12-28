import unittest
import os
import shutil
import sys
# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from lazzaro.core.vector_store import LanceDBVectorStore

class TestLanceDBVectorStore(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_vector_db"
        if os.path.exists(self.test_db):
            shutil.rmtree(self.test_db)
        self.vs = LanceDBVectorStore(db_dir=self.test_db)

    def tearDown(self):
        self.vs.close()
        if os.path.exists(self.test_db):
            shutil.rmtree(self.test_db)

    def test_add_and_search(self):
        nodes = [
            {
                "id": "node_1",
                "content": "I like apples",
                "embedding": [0.1] * 1536,
                "type": "semantic",
                "salience": 0.8,
                "shard_key": "food",
                "timestamp": 1234.5
            },
            {
                "id": "node_2",
                "content": "I hate oranges",
                "embedding": [-0.1] * 1536,
                "type": "semantic",
                "salience": 0.5,
                "shard_key": "food",
                "timestamp": 1235.0
            }
        ]
        self.vs.add(nodes)
        
        # Search for something close to node_1
        results = self.vs.search([0.1] * 1536, limit=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "node_1")

    def test_delete(self):
        nodes = [
            {
                "id": "node_1",
                "content": "test",
                "embedding": [0.1] * 1536,
                "type": "semantic",
                "salience": 0.8,
                "shard_key": "test",
                "timestamp": 1234.5
            }
        ]
        self.vs.add(nodes)
        results = self.vs.search([0.1] * 1536, limit=1)
        self.assertEqual(len(results), 1)
        
        self.vs.delete(["node_1"])
        results = self.vs.search([0.1] * 1536, limit=1)
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main()
