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

class Store(Protocol):
    """
    Protocol for full graph persistence and retrieval.
    Replaces VectorStore in v0.3.0 to handle all nodes, edges, and profile data.
    """
    def add_nodes(self, nodes: List[Dict[str, Any]], user_id: str = "default"):
        """Add a batch of nodes."""
        ...
    
    def get_nodes(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """Retrieve all nodes for a user."""
        ...
    
    def get_latest_version(self) -> int:
        """Get the latest version of the node store."""
        ...
    
    def search_nodes(self, query_emb: List[float], user_id: str = "default", limit: int = 5) -> List[str]:
        """Search for nodes by embedding."""
        ...
    
    def delete_nodes(self, node_ids: List[str], user_id: str = "default"):
        """Delete nodes."""
        ...

    def add_edges(self, edges: List[Dict[str, Any]], user_id: str = "default"):
        """Add a batch of edges."""
        ...

    def delete_edges(self, source_id: Optional[str] = None, user_id: str = "default"):
        """Delete edges, optionally filtered by source."""
        ...

    def get_edges(self, source_id: Optional[str] = None, user_id: str = "default") -> List[Dict[str, Any]]:
        """Retrieve edges, optionally filtered by source."""
        ...

    def save_profile(self, profile_data: Dict[str, Any], user_id: str = "default"):
        """Save user profile state."""
        ...

    def load_profile(self, user_id: str = "default") -> Optional[Dict[str, Any]]:
        """Load user profile state."""
        ...
    
    def close(self):
        """Close connection."""
        ...
