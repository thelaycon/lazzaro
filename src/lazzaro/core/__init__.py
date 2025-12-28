# Core modular components
from .config import MemoryConfig
from .resilience import CircuitBreaker, RetryManager, FallbackManager
from .retriever import MemoryRetriever
from .consolidator import MemoryConsolidator
from .profile_manager import ProfileManager
from .orchestrator import MemoryOrchestrator
from .resilient_providers import create_resilient_providers

# Legacy data structures
from .memory_shard import MemoryShard
from .buffer_graph import BufferGraph
from .profile import Profile
from .query_cache import QueryCache
from .vector_store import LanceDBStore
