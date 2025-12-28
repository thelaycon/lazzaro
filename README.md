# Lazzaro

**Production-Ready Scalable Memory System for AI Agents**

Lazzaro is a high-performance Python library for long-term, structured agent memory using graph-based sharding, hierarchical clustering, and resilient architecture.

## 🚀 Quick Start

```python
from lazzaro import create_lazzaro

# Simple usage with automatic configuration
with create_lazzaro(api_key="your-openai-key") as lazzaro:
    response = lazzaro.chat("I love Rust and distributed systems.")
    insights = lazzaro.get_insights()
    print(insights)
```

## 📦 Installation

```bash
pip install lazzaro
```

### Dependencies
- **Python 3.8+**
- **OpenAI** (for embeddings and LLM)
- **LanceDB** (vector storage)
- **NumPy** (vector operations)
- **NetworkX** (graph operations)

## 🔍 How Lazzaro Works

### 🏗️ Architecture Overview

Lazzaro uses a **4-tier modular architecture**:

```
┌─────────────────────────────────────────────────────────┐
│                    🎛 User Interface (API)          │
├─────────────────────────────────────────────────────────┤
│                  🧠 Memory Orchestrator        │
├─────────────────────────────────────────────────────────┤
│  📦 Retriever │ 🔄 Consolidator │ 👤 Profile Mgr │
├─────────────────────────────────────────────────────────┤
│        🧠 Resilience Providers (LLM + Embedding) │
├─────────────────────────────────────────────────────────┤
│           📊 LanceDB Vector Storage             │
├─────────────────────────────────────────────────────────┤
│         📁 Memory Graph (Shards + Buffer)      │
└─────────────────────────────────────────────────────────┘
```

### 🧠 Memory Processing Flow

#### 1. **Conversation Input**
```python
user_message = "Tell me about your programming preferences"
```

#### 2. **Query Embedding**
```python
# Convert text to vector using resilient provider
query_embedding = embedder.embed_with_retry(message)
# Result: [0.1, -0.3, 0.8, ...]  # 1536-dimensional vector
```

#### 3. **Multi-Stage Retrieval**
```python
# Stage 1: Cache lookup (O(1))
if cached = query_cache.get(query_embedding):
    return cached_results

# Stage 2: Hierarchical search (fast path)
if enable_hierarchy:
    # Search super-nodes first for concept matching
    super_match = find_best_super_node(query_embedding)
    if super_match.similarity > 0.4:
        return super_match.child_nodes[:5]

# Stage 3: Vector similarity search
vector_results = lance_db.search(
    vector=query_embedding,
    where=f"user_id = '{user_id}'",
    limit=5
)
```

#### 4. **Neighbor Boosting**
```python
# Boost related memories through association
for memory in retrieved_memories:
    neighbors = get_connected_nodes(memory.id)
    for neighbor in neighbors:
        neighbor.last_accessed = now()           # Freshness boost
        neighbor.salience *= 1.02               # Importance boost
```

#### 5. **Context Building**
```python
# Build rich context for LLM
context = build_context([
    user_profile.get_insights(),
    retrieved_memories,
    conversation_history[-10:]
])

llm_messages = [
    {"role": "system", "content": "You are an assistant with access to user's memory profile and past conversations."},
    {"role": "system", "content": context},
    {"role": "user", "content": user_message}
]
```

#### 6. **LLM Response**
```python
# Generate response with circuit breaker protection
response = circuit_breaker.call(
    lambda: llm.completion(llm_messages)
)
```

### 📊 Memory Organization

#### **Semantic Sharding**
```python
# Memories organized into topic-based clusters
shards = {
    "work": MemoryShard([
        Node("User works at startup", shard="work"),
        Node("User prefers TypeScript", shard="work"),
        Node("User manages engineering team", shard="work")
    ]),
    "personal": MemoryShard([
        Node("User lives in San Francisco", shard="personal"),
        Node("User enjoys hiking", shard="personal")
    ]),
    "learning": MemoryShard([
        Node("User is learning Rust", shard="learning"),
        Node("User completed ML course", shard="learning")
    ])
}
```

#### **Hierarchical Super-Nodes**
```python
# When shards get large, create abstractions
if len(work_shard.nodes) > 20:
    super_node = SuperNode(
        id="super_work_123",
        content="Topic: work. Contains memories about: User works at startup; manages team; prefers TypeScript",
        embedding=average_embedding(work_shard.nodes),
        child_ids=[node.id for node in work_shard.nodes],
        shard_key="work"
    )
```

#### **Memory Persistence**
```python
# LanceDB storage with B-Tree indexing
db_structure = {
    "nodes.lance": [
        {
            "id": "node_1",
            "user_id": "user_123", 
            "content": "User loves Rust",
            "vector": [0.1, -0.3, 0.8, ...],  # 1536-dim
            "type": "semantic",
            "salience": 0.8,
            "timestamp": 1704067200.0,
            "shard_key": "learning"
        }
    ],
    "edges.lance": [
        {
            "source_id": "node_1",
            "target_id": "node_2", 
            "weight": 0.7,
            "edge_type": "relates_to",
            "user_id": "user_123"
        }
    ],
    "profiles.lance": [
        {
            "user_id": "user_123",
            "data": {
                "preferences": "User prefers low-level languages",
                "knowledge_domains": "systems programming"
            },
            "updated_at": 1704067200.0
        }
    ]
}
```

### 🔄 Memory Consolidation

#### **Fact Extraction**
```python
# Extract atomic facts from conversations
system_prompt = """
Extract distinct, atomic facts from this conversation.
Categorization Guidelines:
1. semantic: Stable facts, preferences, knowledge
2. episodic: Specific events, recent activities  
3. procedural: Processes, workflows, instructions

Format: {"memories": [{"content": "...", "type": "semantic|episodic|procedural", "salience": 0.0-1.0}]}
"""

extracted_facts = llm.completion([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": conversation_history}
])
```

#### **Memory Linking**
```python
# Create associations between memories
for new_fact in extracted_facts:
    # 1. Sequential linking within conversation
    create_edge(new_fact, previous_fact, weight=0.5)
    
    # 2. Semantic similarity linking
    similar_memories = find_similar_memories(new_fact, threshold=0.5)
    for similar in similar_memories:
        create_edge(new_fact, similar, weight=similarity * 0.8)
    
    # 3. Shard-based organization
    shard = infer_shard(new_fact.content)  # "work", "personal", "learning", etc.
    add_to_shard(shard, new_fact)
```

#### **Profile Evolution**
```python
# Update user profile from memory clusters
for connected_component in find_memory_clusters():
    profile_insights = llm.completion([
        {"role": "system", "content": "Extract personality insights from memories"},
        {"role": "user", "content": format_memories(connected_component)}
    ])
    
    # Update profile domains
    profile.update_domain("preferences", profile_insights.preferences)
    profile.update_domain("personality_traits", profile_insights.traits)
```

### 🛡️ Resilience Mechanisms

#### **Circuit Breaker**
```python
# Prevent cascade failures
circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)

try:
    response = circuit_breaker.call(llm.completion, messages)
except CircuitOpenError:
    response = fallback.get_fallback_response("general")
```

#### **Retry with Backoff**
```python
# Handle transient failures
retry_manager = RetryManager(max_retries=3, backoff_factor=2.0)

response = retry_manager.with_retry(
    llm.completion, messages
)
# Retry delays: 1s, 2s, 4s between attempts
```

#### **Query Caching**
```python
# Reduce redundant API calls
cache_key = hash(query_text + user_id)

if cache_key in query_cache:
    return cached_results  # O(1) lookup

# Cache new results
query_cache.set(cache_key, retrieved_ids, ttl=3600)
```

### 📈 Performance Optimization

#### **Vector Indexing**
```python
# B-Tree indexing for efficient user partitioning
# Query: SELECT * FROM nodes WHERE user_id = 'user_123' AND vector <=> query_vec
# Performance: O(log n) user lookup + O(1) vector similarity

# IVF (Inverted File Index) for vector search
# Partition vectors into clusters for faster similarity search
# Performance: Sub-millisecond retrieval for large datasets
```

#### **Batch Operations**
```python
# Minimize API overhead
batch_embeddings = embedder.batch_embed([
    "Memory 1 text", "Memory 2 text", "Memory 3 text"
])  # One API call instead of three

# Bulk database writes
store.add_nodes_batch(new_nodes_data)  # Optimized insert
```

#### **Hierarchical Retrieval**
```python
# Fast path for concept-level queries
if is_conceptual_query(query):
    # Search super-nodes (much smaller dataset)
    super_results = search_super_nodes(query)
    if super_results.similarity > 0.4:
        return super_results.children  # Skip full vector search
```

## 🎯 Usage Examples

### Basic Memory Management
```python
from lazzaro import Lazzaro

lazzaro = Lazzaro(api_key="your-key")

# Chat with memory context
response = lazzaro.chat("What did we discuss about project?")

# Explicitly remember something
lazzaro.remember("User prefers TypeScript over JavaScript", "semantic")

# Search memories
memories = lazzaro.recall("programming languages", limit=5)
for memory in memories:
    print(f"- {memory}")
```

### Advanced Configuration
```python
from lazzaro import Lazzaro
from lazzaro.core.config import MemoryConfig

# Custom configuration
config = MemoryConfig(
    enable_sharding=True,
    max_buffer_size=100,
    consolidate_every=3,
    resilience_settings={
        "max_retries": 5,
        "circuit_breaker_threshold": 10
    }
)

lazzaro = Lazzaro(config=config, api_key="your-key")
```

### Environment Configuration
```bash
# Set environment variables
export LAZZARO_SHARDING=true
export LAZZARO_MAX_BUFFER=100
export LAZZARO_LLM_MODEL=gpt-4o
export LAZZARO_DB_DIR=/data/lazzaro
```

```python
# Load from environment
from lazzaro.core.config import MemoryConfig

config = MemoryConfig.from_env()
lazzaro = Lazzaro(config=config, api_key="your-key")
```

### Multi-Tenant Support
```python
# Switch between users
lazzaro.switch_user("user_123")
user1_insights = lazzaro.get_insights()

lazzaro.switch_user("user_456") 
user2_insights = lazzaro.get_insights()

# Get all users
users = lazzaro.get_all_users()
print(f"Total users: {len(users)}")
```

### Streaming Chat
```python
for chunk in lazzaro.chat_stream("Explain machine learning"):
    if chunk['type'] == 'token':
        print(chunk['content'], end='', flush=True)
    elif chunk['type'] == 'info':
        print(f"\n{chunk['content']}")
```

## 📊 Memory Insights

```python
# Get comprehensive user insights
insights = lazzaro.get_insights()
print(insights)

# Sample output:
"""
1. **Personality Traits**: User shows analytical thinking and preference for structured approaches.
2. **Core Interests & Knowledge**: Deep expertise in distributed systems, Rust programming, and system architecture.
3. **Behavioral Patterns**: User prefers hands-on implementation over theoretical discussions.
4. **Recent Focus**: Currently working on microservices architecture and performance optimization.
"""
```

## 🛡️ Production Features

### Resilience Patterns
- **Circuit Breaker** - Prevents cascade failures
- **Retry with Exponential Backoff** - Handles transient issues
- **Fallback Strategies** - Graceful degradation
- **Comprehensive Error Handling** - Detailed logging and recovery

### Performance Monitoring
```python
# Get system statistics
stats = lazzaro.get_stats()

print(f"Memory nodes: {stats['buffer_nodes']}")
print(f"Retrieval latency: {stats['performance']['avg_retrieval_ms']}ms")
print(f"Cache hit rate: {stats['performance']['cache_hit_rate']}")
print(f"Profile domains: {stats['profile_domains_filled']}/5")
```

### Configuration Management
```yaml
# config.yaml
enable_sharding: true
enable_hierarchy: true
max_buffer_size: 100
consolidate_every: 3
llm_model: "gpt-4o-mini"
db_dir: "/data/lazzaro"
resilience:
  max_retries: 5
  circuit_breaker_threshold: 10
  retry_backoff_factor: 2.0
```

```python
# Load from YAML
config = MemoryConfig.from_file(Path("config.yaml"))
lazzaro = Lazzaro(config=config, api_key="your-key")
```

## 📁 Data Storage

Lazzaro stores data in **LanceDB** for high-performance vector operations:

```
📁 /project/
└── 📁 db/                    # Default database directory
    └── 📁 lancedb/           # LanceDB files
        ├── 📄 nodes.lance    # Memory nodes with embeddings
        ├── 📄 edges.lance    # Memory relationships
        └── 📄 profiles.lance  # User profiles
```

### Multi-Tenant Organization
- **B-Tree indexing** on `user_id` for efficient access
- **User isolation** with secure data separation
- **Scalable storage** supporting thousands of users

## 🎛️ Dashboard

Launch interactive memory visualization dashboard:

```bash
lazzaro-dashboard
```

Features:
- **Live Force-Graph** - Interactive memory visualization
- **User Explorer** - Multi-tenant memory switching
- **Insights Panel** - Full-screen personality analysis
- **Performance Metrics** - Real-time system monitoring

## 🔧 Advanced Usage

### Custom Providers
```python
from lazzaro.core.interfaces import LLMProvider, EmbeddingProvider
from lazzaro import Lazzaro

# Use custom providers
class CustomLLM(LLMProvider):
    def completion(self, messages, response_format=None):
        # Your implementation
        return "Custom response"

lazzaro = Lazzaro(
    llm_provider=CustomLLM(),
    embedding_provider=CustomEmbedder(),
    api_key="not-needed"
)
```

### Background Processing
```python
# Configure async consolidation
config = MemoryConfig(
    enable_async=True,
    consolidate_every=3,
    max_buffer_size=50
)

lazzaro = Lazzaro(config=config, api_key="your-key")

# Consolidation happens automatically in background
```

### Memory Export
```python
# Export memories for analysis
markdown_export = lazzaro.export_memories(format="markdown")
json_export = lazzaro.export_memories(format="json")

# Save to file
with open("memories.md", "w") as f:
    f.write(markdown_export)
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_retriever.py
python -m pytest tests/test_consolidator.py
python -m pytest tests/test_resilience.py

# Run with coverage
python -m pytest --cov=lazzaro tests/
```

## 📈 Performance

### Benchmarks
- **Retrieval Latency**: <50ms (P95 <100ms)
- **Memory Consolidation**: <2s average
- **Cache Hit Rate**: >75% for repeated queries
- **Multi-tenant Scaling**: 10k+ users supported

### Optimization Features
- **Query Caching** - Reduces redundant API calls
- **Batch Embedding** - Minimizes provider overhead
- **Hierarchical Retrieval** - Fast concept lookup
- **Vector Indexing** - LanceDB B-Tree optimization

## 🔒 Security & Privacy

- **Local Storage** - Data stored locally by default
- **User Isolation** - Multi-tenant data separation
- **API Key Management** - Secure credential handling
- **Data Encryption** - Optional at-rest encryption

## 🤝 Integrations

### LangChain
```python
from lazzaro.integrations.langchain_integration import LazzaroLangChainMemory

memory = LazzaroLangChainMemory(api_key="your-key")
# Use with LangChain agents and chains
```

### LangGraph
```python
from lazzaro.integrations.langgraph_integration import LazzaroLangGraph

# Persistent memory for LangGraph workflows
memory = LazzaroLangGraph(api_key="your-key")
```

## 📚 API Reference

### Core Classes
- **Lazzaro** - Main memory system interface
- **MemoryConfig** - Configuration management
- **MemoryOrchestrator** - Advanced orchestration

### Key Methods
- `chat(message)` - Chat with memory context
- `remember(content, type, salience)` - Add explicit memory
- `recall(query, limit)` - Search memories
- `get_insights()` - Get user insights
- `switch_user(user_id)` - Multi-tenant switching

### Configuration Options
- `enable_sharding` - Semantic memory clustering
- `enable_hierarchy` - Super-node creation
- `max_buffer_size` - Short-term memory limit
- `consolidate_every` - Auto-consolidation frequency
- `circuit_breaker_threshold` - Failure tolerance

## 🚀 Production Deployment

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
ENV LAZZARO_DB_DIR=/data
VOLUME ["/data"]

CMD ["python", "-m", "lazzaro.cli.main"]
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lazzaro
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lazzaro
  template:
    spec:
      containers:
      - name: lazzaro
        image: lazzaro:latest
        env:
        - name: LAZZARO_DB_DIR
          value: "/data"
        volumeMounts:
        - name: storage
          mountPath: /data
      volumes:
      - name: storage
        persistentVolume:
          claimName: lazzaro-storage
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LanceDB** - High-performance vector database
- **OpenAI** - Embedding and completion services
- **NetworkX** - Graph algorithms and data structures

---

**Lazzaro** - Production-ready memory systems for AI agents. 🚀