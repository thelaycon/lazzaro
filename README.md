# Lazzaro

**Scalable Memory System Library**

Lazzaro is a Python library designed to give AI agents long-term, scalable, and structured memory. Unlike simple vector databases, Lazzaro uses a **Graph-based** approach combined with **Memory Sharding** and **Hierarchical Clustering** to mimic how human memory works: storing active context in a buffer, consolidating short-term interactions into long-term structures, and forgetting irrelevant details over time.

## Installation

```bash
pip install lazzaro
```

**Optional Dependencies:**
- For Gemini: `pip install google-generativeai`
- For Together AI: `pip install together`
- For LangChain: `pip install langchain-core`
- For Autogen: `pip install pyautogen`

## How It Works

Lazzaro operates on a few core principles to manage memory scalability and relevance:

### 1. Architecture
*   **Sharding**: Memories are automatically categorized into shards (e.g., `work`, `personal`, `health`) based on content.
*   **Deduplication**: Lazzaro automatically merges identical memories during consolidation and ensures retrieval results are diverse and non-redundant.
*   **Profile Evolution**: Beyond facts, Lazzaro extracts and maintains a evolving **User Profile** (preferences, personality, knowledge domains) across multiple fields in a single pass.

### 2. Memory Lifecycle
1.  **Short-Term Memory (STM)**: Every interaction is initially stored in a temporary buffer.
2.  **Consolidation**: Background process extracts atomic facts, updates the graph with new associations, and updates profile insights.
3.  **Forgetting**: Pruning logic removes low-salience associations and archives old nodes to maintain performance.

## Usage

### LLM & Embedding Providers

Lazzaro supports multiple providers out of the box:

```python
from lazzaro.core.providers import OpenAILLM, GeminiLLM, TogetherLLM

# Gemini Example
llm = GeminiLLM(api_key="...", model="gemini-1.5-flash")
embedder = GeminiEmbedder(api_key="...", model="models/embedding-001")

ms = MemorySystem(llm_provider=llm, embedding_provider=embedder)
```

### Framework Integrations

Lazzaro comes with built-in integrations for popular agent frameworks:

#### 🔗 LangChain
```python
from lazzaro.integrations import LazzaroLangChainMemory
from langchain.chains import ConversationChain

memory = LazzaroLangChainMemory(memory_system=ms)
chain = ConversationChain(llm=chat_model, memory=memory)
```

#### 🕸️ LangGraph
```python
from lazzaro.integrations import LazzaroLangGraph

lg = LazzaroLangGraph(ms)
# Add nodes to your graph
builder.add_node("retrieve_memory", lg.get_memory_node())
builder.add_node("record_interaction", lg.get_record_node())
```

#### 🤖 Autogen
```python
from lazzaro.integrations import LazzaroAutogenAgent
from autogen import AssistantAgent

agent = AssistantAgent("assistant", llm_config=...)
LazzaroAutogenAgent(agent, ms) # Hooks memory into the agent
```

#### 🛠️ Google ADK
```python
from lazzaro.integrations import LazzaroADKPlugin

plugin = LazzaroADKPlugin(ms)
# Register retrieval as a tool
agent.add_tool(plugin.as_tool())
```

## Configuration

Lazzaro is highly configurable.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `auto_consolidate` | `True` | Automatically extract facts and update graph after every N conversations. |
| `consolidate_every` | `3` | Frequency of full consolidation runs. |
| `max_buffer_size` | `10` | Maximum number of active nodes in the graph before pruning. |
| `enable_sync` | `True` | Run consolidation tasks in background threads. |
| `enable_sharding` | `True` | Organize memories into semantic shards. |
| `load_from_disk` | `True` | Reload the last saved state from `db/lazzaro.pkl`. |
