"""
Lazzaro - Clean, simplified public interface for the scalable memory system.

This module provides a user-friendly API that hides the complexity of the
underlying modular architecture while maintaining all functionality.
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path

from .core.config import MemoryConfig
from .core.orchestrator import MemoryOrchestrator
from .core.interfaces import LLMProvider, EmbeddingProvider, Store
from .models.graph import Node


class Lazzaro:
    """
    Clean, simplified public interface for Lazzaro memory system.
    
    This class provides a high-level API that abstracts away the complexity
    of the modular architecture while maintaining all functionality.
    
    Example:
        ```python
        # Simple usage with default configuration
        lazzaro = Lazzaro(openai_api_key="your-key")
        response = lazzaro.chat("Tell me about my project preferences")
        insights = lazzaro.get_insights()
        
        # Advanced usage with custom configuration
        config = MemoryConfig.from_file("config.yaml")
        lazzaro = Lazzaro(config=config, openai_api_key="your-key")
        ```
    """
    
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        openai_api_key: Optional[str] = None,
        llm_provider: Optional[LLMProvider] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        store: Optional[Store] = None,
        user_id: str = "default"
    ):
        """
        Initialize Lazzaro memory system.
        
        Args:
            config: Memory configuration. If None, loads from environment variables.
            openai_api_key: OpenAI API key. Required if not using custom providers.
            llm_provider: Custom LLM provider implementation.
            embedding_provider: Custom embedding provider implementation.
            store: Custom storage implementation.
            user_id: User identifier for multi-tenant support.
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = config or MemoryConfig.from_env()
        
        # Initialize orchestrator
        self.orchestrator = MemoryOrchestrator(
            config=self.config,
            openai_api_key=openai_api_key,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            store=store
        )
        
        # Set user ID
        self.orchestrator.user_id = user_id
        
        self.logger.info(f"Lazzaro initialized for user: {user_id}")
    
    def chat(self, message: str) -> str:
        """
        Simple chat interface with memory retrieval.
        
        Args:
            message: User message to process.
            
        Returns:
            LLM response with memory context.
        """
        return self.orchestrator.chat(message)
    
    def chat_stream(self, message: str):
        """
        Streaming chat interface.
        
        Args:
            message: User message to process.
            
        Yields:
            Dictionaries with 'type' ("info" or "token") and 'content'.
        """
        for chunk in self.orchestrator.chat_stream(message):
            yield chunk
    
    def remember(self, content: str, memory_type: str = "semantic", salience: float = 0.5):
        """
        Explicitly add a memory to the system.
        
        Args:
            content: Memory content to store.
            memory_type: Type of memory ("semantic", "episodic", "procedural").
            salience: Importance score (0.0 - 1.0).
        """
        if not self.orchestrator.conversation_active:
            self.orchestrator.start_conversation()
        
        self.orchestrator.add_to_short_term(content, memory_type, salience)
        self.logger.info(f"Added memory: {content[:50]}...")
    
    def recall(self, query: str, limit: int = 5) -> List[str]:
        """
        Search and retrieve relevant memories.
        
        Args:
            query: Search query.
            limit: Maximum number of results.
            
        Returns:
            List of memory contents.
        """
        nodes = self.orchestrator.search_memories(query, limit)
        return [node.content for node in nodes]
    
    def get_insights(self) -> str:
        """
        Get comprehensive user personality and knowledge insights.
        
        Returns:
            Detailed analysis of user patterns, preferences, and knowledge.
        """
        return self.orchestrator.get_insights()
    
    def start_conversation(self) -> str:
        """Start a new conversation session."""
        return self.orchestrator.start_conversation()
    
    def end_conversation(self) -> str:
        """End the current conversation and trigger consolidation."""
        return self.orchestrator.end_conversation()
    
    def switch_user(self, user_id: str):
        """
        Switch to a different user's memory context.
        
        Args:
            user_id: New user identifier.
        """
        self.orchestrator.switch_user(user_id)
        self.logger.info(f"Switched to user: {user_id}")
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary containing performance metrics and system state.
        """
        return self.orchestrator.get_stats()
    
    def run_consolidation(self) -> str:
        """
        Manually trigger deep memory consolidation.
        
        Returns:
            Summary of consolidation actions performed.
        """
        return self.orchestrator.run_consolidation()
    
    def export_memories(self, format: str = "markdown") -> str:
        """
        Export memories in structured format.
        
        Args:
            format: Export format ("markdown" or "json").
            
        Returns:
            Formatted memory export.
        """
        # Simple export implementation - can be enhanced later
        memories = []
        for shard in self.orchestrator.shards.values():
            for node in shard.nodes.values():
                memories.append({
                    "content": node.content,
                    "type": node.type,
                    "salience": node.salience,
                    "category": node.shard_key,
                    "timestamp": node.timestamp
                })
        
        if format == "json":
            import json
            return json.dumps(memories, indent=2, default=str)
        else:
            # Markdown format
            lines = ["# Memory Export\n"]
            for i, mem in enumerate(memories, 1):
                lines.append(f"## {i}. {mem['category']}")
                lines.append(f"**Content:** {mem['content']}")
                lines.append(f"**Importance:** {mem['salience']:.2f}")
                lines.append(f"**Type:** {mem['type']}")
                lines.append("")
            return "\n".join(lines)
    
    def get_profile(self) -> Dict:
        """
        Get user profile data.
        
        Returns:
            Dictionary containing user profile information.
        """
        return self.orchestrator.profile.to_dict()
    
    def close(self):
        """Close the memory system and save state."""
        self.orchestrator.close()
        self.logger.info("Lazzaro memory system closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions for quick usage
def create_lazzaro(
    openai_api_key: str,
    config_file: Optional[Path] = None,
    **kwargs
) -> Lazzaro:
    """
    Convenience function to create Lazzaro instance.
    
    Args:
        openai_api_key: OpenAI API key.
        config_file: Optional path to configuration file.
        **kwargs: Additional configuration parameters.
        
    Returns:
        Configured Lazzaro instance.
    """
    if config_file:
        config = MemoryConfig.from_file(config_file)
    else:
        config = MemoryConfig.from_env()
    
    # Override with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return Lazzaro(config=config, openai_api_key=openai_api_key)


def quick_chat(openai_api_key: str, message: str) -> str:
    """
    Quick one-shot chat with memory.
    
    Args:
        openai_api_key: OpenAI API key.
        message: Message to process.
        
    Returns:
        LLM response with memory context.
    """
    with create_lazzaro(openai_api_key) as lazzaro:
        return lazzaro.chat(message)