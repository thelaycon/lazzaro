"""
Memory Facade - Simplified Abstraction Layer for Lazzaro Memory System

This module provides an intuitive, high-level interface that makes it easy
to work with memories without needing to understand the underlying architecture.
"""

import time
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from .core.config import MemoryConfig
from .core.orchestrator import MemoryOrchestrator


@dataclass
class Memory:
    """Simple memory representation for external use."""
    id: str
    content: str
    importance: float  # 0.0 to 1.0
    category: str
    created_at: str
    last_accessed: str
    access_count: int
    

class MemoryFacade:
    """
    Simple interface for interacting with the Lazzaro memory system.
    
    This class hides the complexity of shards, nodes, edges, and consolidation
    while providing easy-to-use methods for common memory operations.
    
    Example:
        ```python
        memory = MemoryFacade(openai_api_key="your-key")
        
        # Store memories
        memory.remember("I love Python programming", importance=0.8, category="programming")
        memory.remember("Meeting tomorrow at 2 PM", importance=0.9, category="schedule")
        
        # Find memories
        memories = memory.find("Python programming")
        important_memories = memory.get_important_memories(min_importance=0.7)
        
        # Chat with memory
        response = memory.ask("What programming languages do I like?")
        ```
    """
    
    def __init__(self, openai_api_key: str, config: Optional[MemoryConfig] = None):
        """Initialize the memory facade with an OpenAI API key."""
        self.config = config or MemoryConfig.from_env()
        self.orchestrator = MemoryOrchestrator(config=self.config, openai_api_key=openai_api_key)
    
    def remember(self, 
                 content: str, 
                 importance: float = 0.5, 
                 category: str = "general") -> str:
        """
        Store a memory with automatic categorization.
        
        Args:
            content: The memory content to store
            importance: How important this memory is (0.0 to 1.0)
            category: Category for organizing memories (e.g., "work", "personal")
            
        Returns:
            Memory ID for future reference
        """
        if not self.orchestrator.conversation_active:
            self.orchestrator.start_conversation()
        
        memory_id = f"mem_{int(time.time() * 1000)}"
        self.orchestrator.add_to_short_term(content, "semantic", importance)
        
        # Trigger immediate consolidation for important memories
        if importance > 0.8:
            self.orchestrator.end_conversation()
            self.orchestrator.start_conversation()
        
        return memory_id
    
    def find(self, query: str, max_results: int = 5) -> List[Memory]:
        """
        Find memories matching a query.
        
        Args:
            query: Search query text
            max_results: Maximum number of results to return
            
        Returns:
            List of Memory objects matching the query
        """
        nodes = self.orchestrator.search_memories(query, max_results)
        
        memories = []
        for node in nodes:
            memory = Memory(
                id=node.id,
                content=node.content,
                importance=node.salience,
                category=node.shard_key,
                created_at=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(node.timestamp)),
                last_accessed=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(node.last_accessed)),
                access_count=node.access_count
            )
            memories.append(memory)
        
        return memories
    
    def ask(self, question: str) -> str:
        """
        Ask a question with automatic memory retrieval.
        
        Args:
            question: Your question or request
            
        Returns:
            Response with relevant memory context
        """
        return self.orchestrator.chat(question)
    
    def get_important_memories(self, min_importance: float = 0.7) -> List[Memory]:
        """
        Get all memories above a certain importance threshold.
        
        Args:
            min_importance: Minimum importance score (0.0 to 1.0)
            
        Returns:
            List of important Memory objects
        """
        # Get all nodes from buffer
        all_memories = []
        
        for shard in self.orchestrator.shards.values():
            for node in shard.nodes.values():
                if node.salience >= min_importance:
                    memory = Memory(
                        id=node.id,
                        content=node.content,
                        importance=node.salience,
                        category=node.shard_key,
                        created_at=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(node.timestamp)),
                        last_accessed=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(node.last_accessed)),
                        access_count=node.access_count
                    )
                    all_memories.append(memory)
        
        # Sort by importance (descending)
        all_memories.sort(key=lambda m: m.importance, reverse=True)
        return all_memories
    
    def get_memories_by_category(self, category: str) -> List[Memory]:
        """
        Get all memories from a specific category.
        
        Args:
            category: Category name (e.g., "work", "programming", "personal")
            
        Returns:
            List of Memory objects from the specified category
        """
        memories = []
        
        if category in self.orchestrator.shards:
            shard = self.orchestrator.shards[category]
            for node in shard.nodes.values():
                memory = Memory(
                    id=node.id,
                    content=node.content,
                    importance=node.salience,
                    category=node.shard_key,
                    created_at=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(node.timestamp)),
                    last_accessed=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(node.last_accessed)),
                    access_count=node.access_count
                )
                memories.append(memory)
        
        return memories
    
    def get_recent_memories(self, hours: int = 24) -> List[Memory]:
        """
        Get memories from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent Memory objects
        """
        cutoff_time = time.time() - (hours * 3600)
        memories = []
        
        for shard in self.orchestrator.shards.values():
            for node in shard.nodes.values():
                if node.timestamp >= cutoff_time:
                    memory = Memory(
                        id=node.id,
                        content=node.content,
                        importance=node.salience,
                        category=node.shard_key,
                        created_at=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(node.timestamp)),
                        last_accessed=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(node.last_accessed)),
                        access_count=node.access_count
                    )
                    memories.append(memory)
        
        # Sort by creation time (newest first)
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories
    
    def forget(self, memory_id: str) -> bool:
        """
        Remove a specific memory.
        
        Args:
            memory_id: ID of the memory to remove
            
        Returns:
            True if memory was found and removed, False otherwise
        """
        # Find and remove the memory from its shard
        for shard in self.orchestrator.shards.values():
            if memory_id in shard.nodes:
                del shard.nodes[memory_id]
                
                # Remove associated edges
                edges_to_remove = [
                    key for key, edge in shard.edges.items()
                    if edge.source == memory_id or edge.target == memory_id
                ]
                for edge_key in edges_to_remove:
                    del shard.edges[edge_key]
                
                return True
        
        return False
    
    def update_importance(self, memory_id: str, new_importance: float) -> bool:
        """
        Update the importance of a memory.
        
        Args:
            memory_id: ID of the memory to update
            new_importance: New importance score (0.0 to 1.0)
            
        Returns:
            True if memory was found and updated, False otherwise
        """
        for shard in self.orchestrator.shards.values():
            if memory_id in shard.nodes:
                shard.nodes[memory_id].salience = max(0.0, min(1.0, new_importance))
                return True
        
        return False
    
    def get_insights(self) -> Dict:
        """
        Get insights about your memories and patterns.
        
        Returns:
            Dictionary containing insights and statistics
        """
        # Get basic insights from the orchestrator
        insights = self.orchestrator.get_insights()
        
        # Add simplified statistics
        stats = self.orchestrator.get_stats()
        
        return {
            "insights": insights,
            "total_memories": stats["buffer_nodes"],
            "categories": list(self.orchestrator.shards.keys()),
            "recent_conversations": stats["conversation_count"],
            "most_accessed": self._get_most_accessed_memories()
        }
    
    def _get_most_accessed_memories(self, limit: int = 3) -> List[Dict]:
        """Get the most frequently accessed memories."""
        all_memories = []
        
        for shard in self.orchestrator.shards.values():
            for node in shard.nodes.values():
                if node.access_count > 0:
                    all_memories.append({
                        "content": node.content,
                        "access_count": node.access_count,
                        "category": node.shard_key
                    })
        
        all_memories.sort(key=lambda m: m["access_count"], reverse=True)
        return all_memories[:limit]
    
    def consolidate(self) -> str:
        """
        Manually trigger memory consolidation and cleanup.
        
        Returns:
            Summary of consolidation actions performed
        """
        return self.orchestrator.run_consolidation()
    
    def get_categories(self) -> List[str]:
        """
        Get all available memory categories.
        
        Returns:
            List of category names
        """
        return list(self.orchestrator.shards.keys())
    
    def get_summary(self) -> str:
        """
        Get a human-readable summary of your memory state.
        
        Returns:
            Formatted summary text
        """
        stats = self.orchestrator.get_stats()
        categories = self.get_categories()
        
        summary = f"""Memory Summary:
- Total memories: {stats['buffer_nodes']}
- Categories: {', '.join(categories) if categories else 'None'}
- Conversations: {stats['conversation_count']}
- Memory system: {'Active' if stats['conversation_active'] else 'Inactive'}
"""
        
        return summary


# Convenience function for quick usage
def create_memory_facade(openai_api_key: str) -> MemoryFacade:
    """
    Create a new MemoryFacade instance.
    
    Args:
        openai_api_key: Your OpenAI API key
        
    Returns:
        Configured MemoryFacade instance
    """
    return MemoryFacade(openai_api_key)