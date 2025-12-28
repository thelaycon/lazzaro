
import json
import unittest
from unittest.mock import MagicMock, patch
from lazzaro.core.memory_system import MemorySystem
from lazzaro.models.graph import Node, Edge

class TestProfileUpdate(unittest.TestCase):
    def setUp(self):
        self.memory = MemorySystem(
            openai_api_key="dummy",
            llm_provider=MagicMock(),
            embedding_provider=MagicMock(),
            load_from_disk=False,
            auto_consolidate=False
        )
        # Mock embeddings to be deterministic
        self.memory._get_embedding = MagicMock(return_value=[0.1]*10)
        self.memory._batch_embed = MagicMock(return_value=[[0.1]*10]*10)

    def test_multi_field_profile_update(self):
        print("\n--- Testing Multi-Field Profile Update ---")
        
        # 1. Seed the buffer with diverse memories
        memories = [
            "I love programming in Python for data science.",
            "I tend to be very detail-oriented and patient when debugging.",
            "I have 5 years of experience building scalable distributed systems.",
            "I prefer concise, direct communication in meetings."
        ]
        
        for i, content in enumerate(memories):
            node = Node(
                id=f"seed_{i}",
                content=content,
                embedding=[0.1]*10,
                type="semantic",
                shard_key="default"
            )
            # Add to buffer
            self.memory.buffer.add_node(node)

        # Create some edges to form a component (run_consolidation looks for connected components)
        for i in range(len(memories) - 1):
             edge = Edge(source=f"seed_{i}", target=f"seed_{i+1}", weight=0.8)
             self.memory.buffer.add_edge(edge)

        # 2. Mock LLM response for profile extraction
        # Specifically, the call inside _extract_profile_from_contents
        mock_insights = {
            "preferences": "User prefers Python/Data Science and concise communication.",
            "personality_traits": "Detail-oriented and patient.",
            "knowledge_domains": "Experienced in scalable distributed systems.",
            "interaction_style": "Direct communication style."
        }
        
        # We need to find the right mock target. self.memory._call_llm is used.
        # However, _call_llm is also used for chat etc. 
        # For simplicity in this test, we'll patch _extract_profile_from_contents or 
        # just ensure _call_llm returns the expected JSON when called with the right prompt.
        
        def mock_llm_call(messages, response_format=None):
            # Check if it's the profile extraction prompt
            if any("Analyze these related memories" in m['content'] for m in messages):
                return json.dumps(mock_insights)
            return "{}"

        self.memory._call_llm = mock_llm_call

        # 3. Run consolidation
        print("Running consolidation...")
        result = self.memory.run_consolidation()
        print(f"Result: {result}")

        # 4. Assert all mocked fields are updated
        profile_data = self.memory.profile.data
        for domain, expected_insight in mock_insights.items():
            self.assertEqual(profile_data[domain], expected_insight, f"Domain {domain} was not updated correctly")
        
        print("✅ Multi-field profile update verified!")

if __name__ == "__main__":
    unittest.main()
