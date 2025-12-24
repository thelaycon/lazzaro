# Lazzaro

**Scalable Memory System Library**

Lazzaro is a Python library designed to give AI agents long-term, scalable, and structured memory. Unlike simple vector databases, Lazzaro uses a **Graph-based** approach combined with **Memory Sharding** and **Hierarchical Clustering** to mimic how human memory works: storing active context in a buffer, consolidating short-term interactions into long-term structures, and forgetting irrelevant details over time.

## Installation

```bash
pip install lazzaro
```

## How It Works

Lazzaro operates on a few core principles to manage memory scalability and relevance:

### 1. Architecture
*   **Sharding**: Memories are automatically categorized into shards (e.g., `work`, `personal`, `health`) based on content. This allows the system to retrieve only relevant slices of memory, keeping searches fast.
*   **Buffer Graph**: Active memories live in a dynamic graph structure where nodes are facts/thoughts and edges are relationships (associations).
*   **Persistence**: State is automatically persisted to local disk (`db/lazzaro.pkl`) using fast binary serialization.

### 2. Memory Lifecycle
1.  **Short-Term Memory (STM)**: Every user interaction is initially stored in a temporary list.
2.  **Consolidation**: When a conversation ends (or periodically), Lazzaro runs a background process to:
    *   Extract atomic facts from the conversation using an LLM.
    *   Embed these facts and insert them into the appropriate **Shard**.
    *   Link new memories to existing related memories (Graph edges).
3.  **Forgetting**: A buffer limit enforces strict discipline. Old, unused, or low-salience memories are "pruned" (archived/deleted) to keep the active graph lightweight.

### 3. Hierarchy & Super-Nodes
When a shard grows too large, Lazzaro automatically clusters related nodes under a **Super-Node**. This creates a hierarchical index, allowing retrieval to scan high-level topics first before diving into granular details, significantly improving retrieval performance at scale.

## Usage

### CLI (Interactive Mode)

The easiest way to use Lazzaro is via the command-line interface.

```bash
lazzaro-cli
```

**Common Commands:**
*   `/start`: Begin a new conversation session.
*   `/end`: End the current session and trigger background consolidation.
*   `/stats`: View current graph size, cache hit rates, and retrieval latency.
*   `/set <param> <value>`: Update configuration (e.g., `/set max_buffer_size 50`).
*   `/save <filename>`: Export current state to a JSON file.

### Python API

Integrate Lazzaro into your own applications:

```python
from lazzaro import MemorySystem
import os

# Initialize the system
# It will automatically load previous state from db/lazzaro.pkl if it exists
ms = MemorySystem(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    enable_async=True,
    auto_consolidate=True
)

# 1. Start a session
ms.start_conversation()

# 2. Chat with memory context
# The system retrieves relevant memories and injects them into the context
# Use chat_stream to get a streaming response iterator
print("Assistant: ", end="", flush=True)
for token in ms.chat_stream("I'm working on the new physics engine today."):
     if token['type'] == 'token':
         print(token['content'], end="", flush=True)
print()

# 3. Add explicit memories (optional)
ms.add_to_short_term("Project deadline is next Friday.", memory_type="fact")

# 4. End session to trigger consolidation
# This extracts facts, updates the graph, and saves to disk
print(ms.end_conversation())
```

### Framework Integration

#### Using LangChain

Lazzaro allows you to bring your own LLM backend. Here is how to use `ChatOpenAI` (or any other LangChain chat model) as the reasoning engine for Lazzaro.

```python
from lazzaro import MemorySystem
from lazzaro.core.interfaces import LLMProvider
from langchain_openai import ChatOpenAI
from typing import List, Dict

class LangChainAdapter(LLMProvider):
    def __init__(self, model_name: str = "gpt-4"):
        self.model = ChatOpenAI(model=model_name, temperature=0.7)
    
    def completion(self, messages: List[Dict[str, str]], response_format: Dict = None) -> str:
        # 1. Convert Lazzaro messages ({'role': '...', 'content': '...'}) 
        #    to LangChain format if necessary, or pass a simple prompt.
        #    For robust chat, we just use the last user message as the prompt here,
        #    but you could build a full ChatPromptTemplate.
        last_message = messages[-1]['content']
        
        # 2. Handle JSON enforcement if requested (Lazzaro uses this for extraction)
        if response_format and response_format.get("type") == "json_object":
             # In a real app, use .with_structured_output() or prompt engineering
             last_message += "\nIMPORTANT: Return valid JSON only."

        # 3. Invoke the LangChain model
        response = self.model.invoke(last_message)
        return response.content
    
    def completion_stream(self, messages: List[Dict[str, str]], response_format: Dict = None):
         # Implement streaming if desired
         pass

# Initialize Lazzaro with your custom adapter
ms = MemorySystem(
    openai_api_key="...",  # Required for default EmbeddingProvider (unless replaced)
    llm_provider=LangChainAdapter(model_name="gpt-4-turbo"),
    # embedding_provider=MyEmbeddingAdapter()  # Optional: Replace embedder too
)

ms.start_conversation()
print(ms.chat("Hello! I'm using LangChain under the hood."))
```


## Configuration

Lazarus is highly configurable. You can adjust these settings during initialization or via the CLI.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `auto_consolidate` | `True` | Automatically extract facts and update graph after every N conversations. |
| `consolidate_every` | `3` | Frequency of full consolidation runs (in number of conversations). |
| `max_buffer_size` | `10` | Maximum number of active nodes in the graph before older ones are pruned. |
| `enable_async` | `True` | Run consolidation and embedding tasks in background threads for responsiveness. |
| `enable_sharding` | `True` | Organize memories into semantic topics (`work`, `personal`) or date-based shards. |
| `enable_hierarchy` | `True` | Create "Super-Nodes" to summarize large clusters of memories. |
| `load_from_disk` | `True` | Automatically reload the last saved state on initialization. |
