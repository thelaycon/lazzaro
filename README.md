# Lazzaro

**Scalable Memory System Library for AI Agents**

Lazzaro is a high-performance Python library for long-term, structured agent memory using graph-based sharding and hierarchical clustering.

## Installation

```bash
pip install lazzaro
```

*   **Extensions**: `google-generativeai`, `together`, `langchain-core`.

## How it Works

Lazzaro organizes memory into **Semantic Shards** (topic-based Subgraphs) backed by **LanceDB**. It maintains an active **Episodic Buffer** that asynchronously consolidates into the long-term graph, evolving a structured **User Profile** while using **Biological Decay** to prune weak associations.

## Usage

### Basic Initialization
```python
from lazzaro.core.memory_system import MemorySystem

ms = MemorySystem(user_id="alice")
ms.start_conversation()
ms.chat("I love Rust and distributed systems.")
ms.end_conversation()
```

### Multi-Tenant Usage
Lazzaro natively supports B-Tree optimized user partitioning:
```python
# Start as one user
ms = MemorySystem(user_id="user_123")

# Switch context mid-session
ms.switch_user("user_456") 

# Discovery
users = ms.get_all_users()
```

## Advanced APIs

- **`get_insights()`**: LLM-summarized personality/knowledge profile.
- **`export_observations()`**: Export memories as Markdown or JSON.
- **`search_memories(query)`**: Semantic discovery within user context.

## Visual Dashboard

Launch the interactive force-graph explorer:
```bash
lazzaro-dashboard
```

## Integrations

- **LangChain**: `LazzaroLangChainMemory`
- **LangGraph**: `LazzaroLangGraph`
- **CLI**: `lazzaro-cli` for interactive debugging.

## Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `consolidate_every` | `3` | Conv frequency for fact extraction. |
| `max_buffer_size` | `10` | Nodes allowed before archiving. |
| `db_dir` | `"db"` | LanceDB persistence path. |

## License
MIT
