import time
import logging
from typing import List, Dict, Optional, Set
from collections import defaultdict

import numpy as np

from ..models.graph import Node, Edge
from .buffer_graph import BufferGraph
from .interfaces import EmbeddingProvider, LLMProvider, Store
from .memory_shard import MemoryShard
from .query_cache import QueryCache


class MemoryRetriever:
    """Focused on memory search and retrieval logic."""
    
    def __init__(self, config, embedder: EmbeddingProvider, store: Store, 
                 buffer: BufferGraph, shards: Dict[str, MemoryShard],
                 super_nodes: Dict[str, Node], query_cache: Optional[QueryCache] = None):
        self.config = config
        self.embedder = embedder
        self.store = store
        self.buffer = buffer
        self.shards = shards
        self.super_nodes = super_nodes
        self.query_cache = query_cache
        self.logger = logging.getLogger(__name__)
        self.metrics = {"retrieval_times": [], "cache_hits": 0}
    
    def optimized_retrieval(self, query_emb: List[float], query_text: str, 
                           user_id: str = "default") -> List[str]:
        """Perform optimized memory retrieval with caching and hierarchical search."""
        start_time = time.time()
        
        # Check cache first
        if self.query_cache:
            cached = self.query_cache.get_results(query_text)
            if cached:
                self.metrics["cache_hits"] += 1
                return cached
        
        retrieved = []
        
        # 1. Hierarchical Retrieval (Fast path for high-level concepts)
        if self.config.enable_hierarchy and self.super_nodes:
            super_scores = []
            for super_id, super_node in self.super_nodes.items():
                sim = self._cosine_similarity(query_emb, super_node.embedding)
                super_scores.append((super_id, sim))
            
            super_scores.sort(key=lambda x: x[1], reverse=True)
            
            if super_scores and super_scores[0][1] > 0.4:
                best_super = self.super_nodes[super_scores[0][0]]
                for child_id in best_super.child_ids[:10]:
                    child = self.buffer.get_node(child_id)
                    if child and not child.is_super_node:
                        retrieved.append(child_id)
                
                if len(retrieved) >= 5:
                    if self.query_cache:
                        self.query_cache.set_results(query_text, retrieved[:5])
                    return retrieved[:5]
        
        # 2. Vector Store Retrieval (LanceDB)
        limit = 10 if not retrieved else 5
        vector_ids = self.store.search_nodes(query_emb, user_id=user_id, limit=limit)
        
        # Merge results, prioritizing hierarchical if any, then vector
        seen_ids = set(retrieved)
        seen_content = set()
        
        # Deduplicate based on ID first
        final_retrieved = []
        for rid in retrieved:
            node = self.buffer.get_node(rid)
            if node:
                seen_content.add(node.content)
                final_retrieved.append(rid)
        
        for rid in vector_ids:
            if rid not in seen_ids:
                node = self.buffer.get_node(rid)
                if node:
                    # Deduplication by content
                    if node.content not in seen_content:
                        seen_content.add(node.content)
                        final_retrieved.append(rid)
                        seen_ids.add(rid)
        
        final_retrieved = final_retrieved[:5]
        
        # Cache results
        if self.query_cache:
            self.query_cache.set_results(query_text, final_retrieved)
        
        # Track metrics
        retrieval_time = (time.time() - start_time) * 1000
        self.metrics["retrieval_times"].append(retrieval_time)
        
        return final_retrieved
    
    def boost_neighbors(self, retrieved_ids: List[str]):
        """Latency-Aware Boosting: Pull neighbors into the present."""
        neighbors = set()
        for nid in retrieved_ids:
            nbs = self.buffer.get_neighbors(nid)
            neighbors.update(nbs)
        
        count = 0
        for nid in neighbors:
            if nid not in retrieved_ids:
                node = self.buffer.get_node(nid)
                if node:
                    # Inherit freshness
                    node.last_accessed = time.time()
                    # Slight associative boost
                    node.salience = min(1.0, node.salience + 0.02)
                    count += 1
        
        if count > 0:
            self.logger.info(f"Graph: Boosted {count} neighbor nodes via association")
    
    def get_relevant_shards(self, query: str, max_shards: int = 3) -> List[str]:
        """Get most relevant shards for a query."""
        if not self.config.enable_sharding or not self.shards:
            return ["default"]
        
        # Optimization: If few shards exist, search all to avoid missing relevant info
        if len(self.shards) <= 5:
            return list(self.shards.keys())
        
        shard_scores = []
        for shard_key, shard in self.shards.items():
            hours_since_access = (time.time() - shard.last_accessed) / 3600
            recency_score = 1.0 / (1.0 + hours_since_access)
            size_score = min(1.0, len(shard.nodes) / 100)
            combined_score = recency_score * 0.7 + size_score * 0.3
            shard_scores.append((shard_key, combined_score))
        
        shard_scores.sort(key=lambda x: x[1], reverse=True)
        return [key for key, _ in shard_scores[:max_shards]]
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate normalized cosine similarity between two vectors."""
        if not v1 or not v2:
            return 0.0
        a, b = np.array(v1), np.array(v2)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / norm) if norm > 0 else 0.0
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching."""
        if self.query_cache:
            cached = self.query_cache.get_embedding(text)
            if cached:
                return cached
        
        embedding = self.embedder.embed(text)
        if self.query_cache:
            self.query_cache.set_embedding(text, embedding)
        return embedding
    
    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Handle batch embedding requests to minimize provider overhead."""
        if not texts:
            return []
        return self.embedder.batch_embed(texts)
    
    def get_stats(self) -> Dict:
        """Get retrieval performance statistics."""
        avg_retrieval = (
            np.mean(self.metrics["retrieval_times"])
            if self.metrics["retrieval_times"]
            else 0
        )
        p95_retrieval = (
            np.percentile(self.metrics["retrieval_times"], 95)
            if self.metrics["retrieval_times"]
            else 0
        )
        cache_hit_rate = (
            self.query_cache.get_hit_rate() if self.query_cache else 0.0
        )
        
        return {
            "avg_retrieval_ms": f"{avg_retrieval:.1f}",
            "p95_retrieval_ms": f"{p95_retrieval:.1f}",
            "cache_hit_rate": f"{cache_hit_rate:.1%}",
            "total_retrievals": len(self.metrics["retrieval_times"]),
            "cache_hits": self.metrics["cache_hits"]
        }