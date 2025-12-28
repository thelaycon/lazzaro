"""
Simple Memory Abstraction Layer Example

This file demonstrates how to use the MemoryFacade class for easy memory management.
"""

from lazzaro import MemoryFacade, create_memory_facade


def main():
    """Example usage of the MemoryFacade abstraction layer."""
    
    # Initialize with your OpenAI API key
    memory = create_memory_facade("your-openai-api-key-here")
    
    # === Storing Memories ===
    # Store memories with different importance levels
    memory.remember("I love Python programming", importance=0.9, category="programming")
    memory.remember("Meeting with team tomorrow at 2 PM", importance=0.8, category="schedule")
    memory.remember("User prefers dark theme in IDEs", importance=0.6, category="preferences")
    memory.remember("Working on machine learning project", importance=0.7, category="projects")
    
    print("✓ Stored 4 memories")
    
    # === Finding Memories ===
    # Search for memories
    python_memories = memory.find("Python")
    print(f"\nFound {len(python_memories)} Python-related memories:")
    for mem in python_memories:
        print(f"  - {mem.content} (Importance: {mem.importance:.2f})")
    
    # Get important memories
    important_memories = memory.get_important_memories(min_importance=0.7)
    print(f"\n{len(important_memories)} important memories:")
    for mem in important_memories:
        print(f"  - {mem.content} [{mem.category}]")
    
    # === Chat with Memory Context ===
    response = memory.ask("What programming languages do I like?")
    print(f"\nChat Response: {response}")
    
    # === Memory Organization ===
    categories = memory.get_categories()
    print(f"\nMemory categories: {categories}")
    
    programming_memories = memory.get_memories_by_category("programming")
    print(f"Programming memories: {len(programming_memories)}")
    
    # === Insights ===
    insights = memory.get_insights()
    print(f"\nMemory Insights:")
    print(f"  Total memories: {insights['total_memories']}")
    print(f"  Categories: {', '.join(insights['categories'])}")
    
    # === Summary ===
    print(f"\n{memory.get_summary()}")
    
    # === Memory Management ===
    # Update importance of a memory
    if python_memories:
        memory.update_importance(python_memories[0].id, 0.95)
        print(f"\n✓ Updated importance of Python memory to 0.95")
    
    # Get recent memories (last 24 hours)
    recent = memory.get_recent_memories(hours=24)
    print(f"Recent memories (24h): {len(recent)}")


class SimpleMemoryManager:
    """
    Even simpler wrapper for absolute beginners.
    
    This class provides the most basic operations with sensible defaults.
    """
    
    def __init__(self, api_key: str):
        self.facade = create_memory_facade(api_key)
    
    def add(self, text: str, category: str = "general"):
        """Add a memory with default importance."""
        return self.facade.remember(text, importance=0.7, category=category)
    
    def search(self, query: str):
        """Search memories."""
        return self.facade.find(query)
    
    def ask(self, question: str):
        """Chat with memory context."""
        return self.facade.ask(question)
    
    def summary(self):
        """Get a quick summary."""
        return self.facade.get_summary()


def simple_example():
    """Ultra-simple usage example."""
    manager = SimpleMemoryManager("your-api-key")
    
    # Add some memories
    manager.add("I prefer coffee over tea", "preferences")
    manager.add("Working on web application", "projects")
    
    # Search
    results = manager.search("web")
    print(f"Found {len(results)} results")
    
    # Ask question
    answer = manager.ask("What am I working on?")
    print(f"Answer: {answer}")
    
    # Summary
    print(manager.summary())


if __name__ == "__main__":
    print("=== Lazzaro Memory Facade Examples ===\n")
    
    print("1. Full Features Example:")
    print("Run main() to see all features\n")
    
    print("2. Simple Example:")
    print("Run simple_example() for basic usage\n")
    
    print("Note: Replace 'your-openai-api-key-here' with your actual API key")