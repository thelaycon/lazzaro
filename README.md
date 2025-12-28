# Lazzaro

**Simple, Scalable Memory System for AI Agents**

Lazzaro provides an intuitive interface for AI agents to store, retrieve, and reason over long-term memories with automatic semantic organization.

## 🚀 Quick Start

```python
from lazzaro.memory_facade import create_memory_facade

# Initialize with your OpenAI API key
memory = create_memory_facade("your-openai-api-key")

# Store memories
memory.remember("I love Python programming", importance=0.9, category="programming")
memory.remember("Meeting tomorrow at 2 PM", importance=0.8, category="schedule")

# Chat with automatic memory retrieval
response = memory.ask("What programming languages do I like?")
print(response)  # Automatically uses relevant memories

# Search memories
python_memories = memory.find("Python")
for mem in python_memories:
    print(f"- {mem.content} (Importance: {mem.importance:.2f})")
```

## 📦 Installation

```bash
pip install lazzaro
```

**Requirements:**
- Python 3.8+
- OpenAI API key
- (Dependencies auto-installed: LanceDB, NumPy, NetworkX)

## 💡 Key Features

### 🧠 Simple Memory Management
```python
# Store memories with automatic categorization
memory.remember("User prefers dark theme", importance=0.7, category="preferences")
memory.remember("Working on ML project", importance=0.8, category="projects")
```

### 🔍 Smart Retrieval
```python
# Semantic search
results = memory.find("machine learning")

# Get important memories
important = memory.get_important_memories(min_importance=0.7)

# Get memories by category
work_memories = memory.get_memories_by_category("work")
```

### 💬 Conversational Memory
```python
# Automatic memory retrieval during conversations
response = memory.ask("What are my current projects?")
# System finds relevant memories and includes them in LLM context
```

### 📊 Insights & Analytics
```python
# Get personality and pattern insights
insights = memory.get_insights()

# Get memory statistics
summary = memory.get_summary()
print(summary)
# Output:
# Memory Summary:
# - Total memories: 15
# - Categories: programming, schedule, preferences, projects
# - Conversations: 3
# - Memory system: Active
```

## 🎯 Usage Examples

### Ultra-Simple Interface
```python
from examples.memory_facade_examples import SimpleMemoryManager

manager = SimpleMemoryManager("your-api-key")

# Add memories
manager.add("I prefer coffee over tea", "preferences")
manager.add("Working on web application", "projects")

# Chat with memory
answer = manager.ask("What am I working on?")
print(answer)
```

### Advanced Memory Management
```python
# Update importance
memory.update_importance(memory_id, 0.95)

# Remove memories
memory.forget(memory_id)

# Get recent memories (last 24 hours)
recent = memory.get_recent_memories(hours=24)

# Manual consolidation
memory.consolidate()
```

### Multi-Category Organization
```python
# Get all categories
categories = memory.get_categories()
print(f"Categories: {categories}")

# Access memories by category
for category in categories:
    memories = memory.get_memories_by_category(category)
    print(f"{category}: {len(memories)} memories")
```

## 🏗️ Architecture Overview

Lazzaro uses a layered architecture to provide both simplicity and power:

```
┌─────────────────────────────────────┐
│      Memory Facade (Simple API)     │  ← You are here
├─────────────────────────────────────┤
│      Lazzaro Core Interface         │  ← Advanced usage
├─────────────────────────────────────┤
│   Memory Orchestrator & Components │  ← Internal magic
├─────────────────────────────────────┤
│         Vector Storage (LanceDB)    │  ← Persistence layer
└─────────────────────────────────────┘
```

### What Happens Behind the Scenes

1. **Memory Storage**: Content gets embedded and stored with semantic categorization
2. **Automatic Organization**: Memories cluster into topics (work, personal, etc.)
3. **Smart Retrieval**: Semantic search + association boosting for relevant results
4. **Conversation Context**: Relevant memories automatically included in LLM prompts
5. **Profile Learning**: System learns patterns and preferences over time

## 🔧 Configuration

### Environment Variables
```bash
export LAZZARO_LLM_MODEL=gpt-4o-mini
export LAZZARO_DB_DIR=./my_memories
export LAZZARO_SHARDING=true
```

### Custom Configuration
```python
from lazzaro import Lazzaro, MemoryConfig

config = MemoryConfig(
    max_buffer_size=50,
    consolidate_every=3,
    enable_sharding=True
)

lazzaro = Lazzaro(config=config, openai_api_key="your-key")
```

## 📁 Data Storage

Memories are stored locally in LanceDB format:

```
📁 project/
└── 📁 db/
    ├── 📄 nodes.lance     # Memory content & embeddings
    ├── 📄 edges.lance     # Memory relationships
    └── 📄 profiles.lance  # User insights & patterns
```

## 🚀 Production Features

### Performance
- **Sub-50ms retrieval** for typical queries
- **Automatic caching** for repeated searches
- **Batch processing** for efficiency
- **Hierarchical indexing** for scalability

### Reliability
- **Circuit breaker** prevents API failures
- **Retry with backoff** handles network issues
- **Graceful degradation** when services are unavailable
- **Local-first storage** for data privacy

### Multi-User Support
```python
# Switch between users
memory.switch_user("user_123")
user1_memories = memory.find("preferences")

memory.switch_user("user_456")
user2_memories = memory.find("preferences")
```

## 📊 Memory Object Structure

All memory operations return `Memory` objects:

```python
@dataclass
class Memory:
    id: str           # Unique identifier
    content: str      # Memory text
    importance: float # 0.0 to 1.0 importance score
    category: str     # Auto-categorized topic
    created_at: str   # Creation timestamp
    last_accessed: str # Last access time
    access_count: int # Access frequency
```

## 🎖️ Importance Guidelines

Use importance levels to prioritize memories:

- **0.9-1.0**: Critical (deadlines, passwords, key facts)
- **0.7-0.8**: Important (preferences, work details)
- **0.5-0.6**: General information (context, casual facts)
- **0.3-0.4**: Contextual details (minor info)
- **0.1-0.2**: Temporary (quick notes, fleeting thoughts)

## 📚 Advanced Usage

### Streaming Chat
```python
for chunk in memory.ask_stream("Tell me about your projects"):
    if chunk['type'] == 'token':
        print(chunk['content'], end='', flush=True)
```

### Memory Export
```python
# Export for analysis
markdown = memory.export_memories("markdown")
json_data = memory.export_memories("json")
```

### Custom Providers
```python
# Use your own LLM/embedding providers
from lazzaro import Lazzaro
from lazzaro.core.interfaces import LLMProvider

class CustomLLM(LLMProvider):
    def completion(self, messages, response_format=None):
        # Your implementation
        return "Custom response"

memory = Lazzaro(llm_provider=CustomLLM())
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_memory_facade.py
python -m pytest tests/test_retriever.py
```

## 📈 Performance Benchmarks

- **Retrieval Latency**: <50ms (P95 <100ms)
- **Memory Consolidation**: <2s average
- **Cache Hit Rate**: >75% for repeated queries
- **Storage Efficiency**: ~1KB per memory (with embedding)

## 🔒 Privacy & Security

- **Local Storage**: Data stored locally by default
- **User Isolation**: Complete data separation between users
- **API Key Safety**: Secure credential management
- **Data Portability**: Easy export and migration

## 🤝 Integrations

### LangChain Integration
```python
from lazzaro.integrations.langchain_integration import LazzaroLangChainMemory

memory = LazzaroLangChainMemory(api_key="your-key")
# Use with LangChain agents and chains
```

### LangGraph Integration
```python
from lazzaro.integrations.langgraph_integration import LazzaroLangGraph

memory = LazzaroLangGraph(api_key="your-key")
# Persistent memory for LangGraph workflows
```

## 📖 API Reference

### MemoryFacade Main Methods
- `remember(content, importance, category)` - Store memory
- `find(query, max_results)` - Search memories
- `ask(question)` - Chat with memory context
- `get_important_memories(min_importance)` - Get high-importance memories
- `get_memories_by_category(category)` - Category-based retrieval
- `get_recent_memories(hours)` - Time-based retrieval
- `get_insights()` - Personality and pattern analysis
- `get_summary()` - System overview
- `update_importance(memory_id, new_importance)` - Modify memory importance
- `forget(memory_id)` - Remove memory
- `consolidate()` - Manual memory optimization

### Utility Functions
- `get_categories()` - List all memory categories
- `switch_user(user_id)` - Multi-tenant switching

## 🚀 Deployment

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
CMD ["python", "-m", "lazzaro.cli.main"]
```

### Environment Configuration
```bash
# Production settings
export LAZZARO_LLM_MODEL=gpt-4o
export LAZZARO_DB_DIR=/data/lazzaro
export LAZZARO_MAX_BUFFER=100
export LAZZARO_SHARDING=true
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Lazzaro** - Making AI memory simple and powerful. 🚀

For detailed documentation and examples, see [MEMORY_FACADE_README.md](MEMORY_FACADE_README.md).