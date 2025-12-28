import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from lazzaro.core.memory_system import MemorySystem
from lazzaro.core.interfaces import LLMProvider, EmbeddingProvider
from lazzaro.core.providers import OpenAILLM

class MockLLM(LLMProvider):
    def completion(self, messages, response_format=None):
        return "Mock response"

class MockEmbedder(EmbeddingProvider):
    def embed(self, text):
        return [0.1] * 1536
    
    def batch_embed(self, texts):
        return [[0.1] * 1536 for _ in texts]

class TestProviders(unittest.TestCase):
    def test_custom_providers(self):
        llm = MockLLM()
        embedder = MockEmbedder()
        
        ms = MemorySystem(
            openai_api_key="fake",
            enable_async=False,
            load_from_disk=False, # Don't load persistence
            llm_provider=llm,
            embedding_provider=embedder
        )
        
        # Test LLM injection
        ms.start_conversation()
        response = ms.chat("Hello")
        self.assertEqual(response, "Mock response")
        
        # Test Embedder injection (implicitly tested via chat calling _get_embedding)
        # We can also check internal state if we exposed it, or trust chat ran without error.
        
    @patch('lazzaro.core.providers.openai')
    def test_default_providers(self, mock_openai):
        # Setup OpenAI mocks
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "OpenAI response"
        mock_client.chat.completions.create.return_value = mock_completion
        
        mock_emb_resp = MagicMock()
        mock_emb_resp.data[0].embedding = [0.2] * 1536
        mock_client.embeddings.create.return_value = mock_emb_resp

        ms = MemorySystem(
            openai_api_key="fake",
            enable_async=False, 
            load_from_disk=False
        )
        
        ms.start_conversation()
        response = ms.chat("Hello")
        
        self.assertEqual(response, "OpenAI response")
        # Verify OpenAI client was called
        mock_client.chat.completions.create.assert_called()
        mock_client.embeddings.create.assert_called()

if __name__ == '__main__':
    unittest.main()
