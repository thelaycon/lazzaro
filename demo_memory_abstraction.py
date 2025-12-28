#!/usr/bin/env python3
"""
Lazzaro Memory Abstraction Demo

This script demonstrates the simplified memory interface concepts.
Run this to see how easy memory management can be with Lazzaro.
"""

def demo_memory_facade():
    """Demonstrate the simplified memory interface."""
    
    print("=== Lazzaro Memory Facade Demo ===\n")
    
    print("1. MEMORY STORAGE")
    print("   # Store memories with automatic categorization")
    print("   memory.remember('I love Python programming', importance=0.9, category='programming')")
    print("   memory.remember('Meeting tomorrow at 2 PM', importance=0.8, category='schedule')")
    print()
    
    print("2. MEMORY RETRIEVAL")
    print("   # Semantic search")
    print("   python_memories = memory.find('Python')")
    print("   for mem in python_memories:")
    print("       print(f'- {mem.content} (Importance: {mem.importance:.2f})')")
    print()
    
    print("3. CONVERSATIONAL MEMORY")
    print("   # Automatic memory retrieval during chat")
    print("   response = memory.ask('What programming languages do I like?')")
    print("   # System automatically finds relevant Python memories!")
    print()
    
    print("4. ORGANIZATION & INSIGHTS")
    print("   # Get memories by category")
    print("   work_memories = memory.get_memories_by_category('programming')")
    print("   # Get important memories")
    print("   important = memory.get_important_memories(min_importance=0.7)")
    print("   # Get insights about your patterns")
    print("   insights = memory.get_insights()")
    print()
    
    print("5. MEMORY MANAGEMENT")
    print("   # Update importance")
    print("   memory.update_importance(memory_id, 0.95)")
    print("   # Remove memories")
    print("   memory.forget(memory_id)")
    print("   # Get recent memories")
    print("   recent = memory.get_recent_memories(hours=24)")
    print()
    
    print("=== REAL-WORLD USAGE ===\n")
    
    print("# Example 1: Personal Assistant")
    print("""
assistant = create_memory_facade(api_key)

# Learn user preferences
assistant.remember("User prefers dark mode in all apps", importance=0.8, category="preferences")
assistant.remember("User is allergic to peanuts", importance=1.0, category="health")
assistant.remember("User works 9-5 Mon-Fri", importance=0.7, category="schedule")

# Smart responses
response = assistant.ask("What should I know about the user's preferences?")
# Automatically includes: dark mode preference, work schedule
""")
    
    print("# Example 2: Project Management")
    print("""
pm = create_memory_facade(api_key)

# Track project details
pm.remember("Project Alpha uses React and Node.js", importance=0.9, category="technology")
pm.remember("Client prefers weekly status meetings", importance=0.8, category="communication")
pm.remember("Deadline: December 15th", importance=0.95, category="deadlines")

# Project context
status = pm.ask("What are the key details about Project Alpha?")
# Returns technology stack, communication preferences, and deadline
""")
    
    print("# Example 3: Learning Companion")
    print("""
tutor = create_memory_facade(api_key)

# Track learning progress
tutor.remember("User is learning Rust and finding ownership challenging", importance=0.8, category="learning")
tutor.remember("User has Python background", importance=0.7, category="background")
tutor.remember("User prefers hands-on coding over theory", importance=0.9, category="style")

# Personalized help
help_text = tutor.ask("How should I explain Rust ownership to someone with Python background?")
# Uses Python knowledge and hands-on preference in response
""")
    
    print("=== BENEFITS ===\n")
    print("✅ SIMPLICITY: No need to understand nodes, edges, or shards")
    print("✅ AUTOMATIC: Smart categorization and importance management")
    print("✅ CONTEXTUAL: Automatic memory retrieval during conversations")
    print("✅ SCALABLE: Handles thousands of memories efficiently")
    print("✅ INSIGHTFUL: Learns patterns and preferences over time")
    print()
    
    print("=== GET STARTED ===\n")
    print("# Install")
    print("pip install lazzaro")
    print()
    print("# Basic usage")
    print("from lazzaro.memory_facade import create_memory_facade")
    print("memory = create_memory_facade('your-openai-api-key')")
    print("memory.remember('I love Python programming', importance=0.9)")
    print("response = memory.ask('What programming languages do I like?')")
    print()
    print("That's it! 🚀")


if __name__ == "__main__":
    demo_memory_facade()