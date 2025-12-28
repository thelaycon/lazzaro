# New modular API
from .api import Lazzaro, create_lazzaro, quick_chat

# Simple memory facade (when dependencies are available)
try:
    from .memory_facade import MemoryFacade, Memory, create_memory_facade
except ImportError:
    # Facade requires optional dependencies
    MemoryFacade = None
    Memory = None
    create_memory_facade = None

# Core components for advanced usage
from .core.config import MemoryConfig
from .core.orchestrator import MemoryOrchestrator
from .core.resilience import CircuitBreaker, RetryManager, FallbackManager
from .core.resilient_providers import create_resilient_providers

# Data structures
from .models.graph import Node, Edge
from .core.memory_shard import MemoryShard
from .core.buffer_graph import BufferGraph
from .core.profile import Profile
from .core.query_cache import QueryCache
from .core.vector_store import LanceDBStore
