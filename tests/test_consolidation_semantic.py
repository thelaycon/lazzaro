import unittest
from unittest.mock import MagicMock, patch
import json
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from lazzaro.core.memory_system import MemorySystem
from lazzaro.models.graph import Node

class TestSemanticConsolidation(unittest.TestCase):
    def setUp(self):
        # Mock LLM and Embedder
        self.mock_llm = MagicMock()
        self.mock_embedder = MagicMock()
        
        self.ms = MemorySystem(
            llm_provider=self.mock_llm,
            embedding_provider=self.mock_embedder,
            enable_async=False,
            load_from_disk=False
        )

    def test_semantic_merging(self):
        # 1. Setup mocks for first fact
        fact1_content = "User prefers Python for data science"
        fact1_emb = [0.1] * 1536
        
        self.mock_llm.completion.return_value = json.dumps({
            "memories": [{"content": fact1_content, "type": "semantic", "salience": 0.8, "topic": "work"}]
        })
        self.mock_embedder.batch_embed.return_value = [fact1_emb]
        
        # 2. Add some initial context to trigger consolidation
        self.ms.start_conversation()
        self.ms.add_to_short_term("I like Python for DS")
        self.ms.end_conversation()
        
        # Verify one node exists
        nodes_count, _ = self.ms.buffer.size()
        self.assertEqual(nodes_count, 1)
        node_id = list(self.ms.buffer.nodes.keys())[0]
        self.assertEqual(self.ms.buffer.nodes[node_id].content, fact1_content)
        
        # 3. Setup mocks for second, semantically similar fact
        fact2_content = "The user has a preference for the Python programming language in data science tasks"
        # High similarity embedding (not exactly the same but very close for cosine sim)
        fact2_emb = [0.1001] * 1536
        
        self.mock_llm.completion.return_value = json.dumps({
            "memories": [{"content": fact2_content, "type": "semantic", "salience": 0.9, "topic": "work"}]
        })
        self.mock_embedder.batch_embed.return_value = [fact2_emb]
        
        # 4. End second conversation
        self.ms.start_conversation()
        self.ms.add_to_short_term("Python is my go-to for data work")
        self.ms.end_conversation()
        
        # Verify that still only ONE node exists due to semantic merging
        final_nodes_count, _ = self.ms.buffer.size()
        self.assertEqual(final_nodes_count, 1, "Should have merged semantically similar facts")
        
        # Verify salience was updated (it will be decayed by 0.01 at the end of end_conversation)
        updated_node = self.ms.buffer.get_node(node_id)
        # 0.9 decayed once: 0.2 + (0.9 - 0.2) * 0.99 = 0.893
        self.assertAlmostEqual(updated_node.salience, 0.893, places=3)
        self.assertEqual(updated_node.access_count, 1) # Internal consolidation merge boosts count

if __name__ == '__main__':
    unittest.main()
