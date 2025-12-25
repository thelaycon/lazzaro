from typing import List, Dict, Any, Protocol, Optional

class LLMProvider(Protocol):
    """
    Protocol for LLM interactions.
    
    Any class implementing this protocol can be used as the logic engine in MemorySystem.

    Example:
        ```python
        class MyLLM(LLMProvider):
            def completion(self, messages, response_format=None):
                return "Simulated response"
        ```
    """
    def completion(self, messages: List[Dict[str, str]], response_format: Dict = None) -> str:
        """
        Generate a completion for the given messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            response_format: Optional dict to specify output format (e.g. {"type": "json_object"}).
        """
        ...
    
    def completion_stream(self, messages: List[Dict[str, str]], response_format: Dict = None):
        """
        Generate a streaming completion for the given messages.
        Should yield chunks of the response content as strings.
        """
        ...

class EmbeddingProvider(Protocol):
    """
    Protocol for text embedding.
    
    Any class implementing this protocol can be used to generate vector representations
    for semantic retrieval in MemorySystem.

    Example:
        ```python
        class MyEmbedder(EmbeddingProvider):
            def embed(self, text):
                return [0.1] * 1536
        ```
    """
    def embed(self, text: str) -> List[float]:
        """Embed a single string."""
        ...
    
    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings."""
        ...
