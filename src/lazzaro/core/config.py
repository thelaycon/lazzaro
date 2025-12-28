from dataclasses import dataclass, field
from typing import Optional
import os
from pathlib import Path


@dataclass
class MemoryConfig:
    """Centralized configuration for Lazzaro memory system."""
    
    # Core feature flags
    enable_sharding: bool = True
    enable_hierarchy: bool = True
    enable_caching: bool = True
    enable_async: bool = True
    
    # Thresholds and limits
    max_shard_size: int = 500
    super_node_threshold: int = 20
    max_buffer_size: int = 10
    prune_threshold: float = 0.5
    
    # Timing and frequency
    consolidate_every: int = 3
    decay_rate: float = 0.01
    
    # Provider settings
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    
    # Storage
    db_dir: str = "db"
    load_from_disk: bool = True
    
    # Resilience settings
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    
    # Performance tuning
    cache_size: int = 1000
    batch_size: int = 10
    
    @classmethod
    def from_file(cls, config_path: Path) -> "MemoryConfig":
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(config_path) as f:
                data = yaml.safe_load(f)
            return cls(**data)
        except ImportError:
            raise ImportError("PyYAML required for config file loading. Install with: pip install pyyaml")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Load configuration from environment variables."""
        return cls(
            enable_sharding=os.getenv("LAZZARO_SHARDING", "true").lower() == "true",
            enable_hierarchy=os.getenv("LAZZARO_HIERARCHY", "true").lower() == "true",
            enable_caching=os.getenv("LAZZARO_CACHING", "true").lower() == "true",
            enable_async=os.getenv("LAZZARO_ASYNC", "true").lower() == "true",
            max_shard_size=int(os.getenv("LAZZARO_MAX_SHARD_SIZE", "500")),
            super_node_threshold=int(os.getenv("LAZZARO_SUPER_NODE_THRESHOLD", "20")),
            max_buffer_size=int(os.getenv("LAZZARO_MAX_BUFFER_SIZE", "10")),
            prune_threshold=float(os.getenv("LAZZARO_PRUNE_THRESHOLD", "0.5")),
            consolidate_every=int(os.getenv("LAZZARO_CONSOLIDATE_EVERY", "3")),
            decay_rate=float(os.getenv("LAZZARO_DECAY_RATE", "0.01")),
            llm_model=os.getenv("LAZZARO_LLM_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("LAZZARO_EMBEDDING_MODEL", "text-embedding-3-small"),
            db_dir=os.getenv("LAZZARO_DB_DIR", "db"),
            load_from_disk=os.getenv("LAZZARO_LOAD_FROM_DISK", "true").lower() == "true",
            max_retries=int(os.getenv("LAZZARO_MAX_RETRIES", "3")),
            retry_backoff_factor=float(os.getenv("LAZZARO_RETRY_BACKOFF", "2.0")),
            circuit_breaker_threshold=int(os.getenv("LAZZARO_CIRCUIT_THRESHOLD", "5")),
            circuit_breaker_timeout=int(os.getenv("LAZZARO_CIRCUIT_TIMEOUT", "60")),
            cache_size=int(os.getenv("LAZZARO_CACHE_SIZE", "1000")),
            batch_size=int(os.getenv("LAZZARO_BATCH_SIZE", "10")),
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }