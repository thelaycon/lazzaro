import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path if not installed yet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from lazzaro.models.graph import Node, Edge
from lazzaro.core.memory_system import MemorySystem

class TestLazzaroBasic(unittest.TestCase):
    def test_node_creation(self):
        node = Node(id="test_1", content="Hello world")
        self.assertEqual(node.id, "test_1")
        self.assertEqual(node.content, "Hello world")
        self.assertEqual(node.type, "semantic")

    def test_edge_creation(self):
        edge = Edge(source="a", target="b")
        self.assertEqual(edge.source, "a")
        self.assertEqual(edge.target, "b")
        self.assertEqual(edge.weight, 1.0)

    @patch('lazzaro.core.memory_system.openai')
    def test_memory_system_init(self, mock_openai):
        ms = MemorySystem(openai_api_key="fake-key", enable_async=False)
        self.assertIsNotNone(ms)
        self.assertEqual(ms.model, "gpt-4o-mini")
        self.assertTrue(ms.enable_sharding)

if __name__ == '__main__':
    unittest.main()
