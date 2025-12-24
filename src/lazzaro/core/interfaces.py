from typing import List, Dict, Any, Protocol, Optional

class LLMProvider(Protocol):
    """Protocol for LLM interactions."""
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
    """Protocol for text embedding."""
    def embed(self, text: str) -> List[float]:
        """Embed a single string."""
        ...
    
    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings."""
        ...
