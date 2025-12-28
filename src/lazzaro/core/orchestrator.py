import time
import logging
from typing import Dict, List, Optional

from .config import MemoryConfig
from .interfaces import LLMProvider, EmbeddingProvider, Store
from .resilient_providers import create_resilient_providers
from .retriever import MemoryRetriever
from .consolidator import MemoryConsolidator
from .profile_manager import ProfileManager
from .buffer_graph import BufferGraph
from .memory_shard import MemoryShard
from .profile import Profile
from .query_cache import QueryCache
from .vector_store import LanceDBStore
from ..models.graph import Node, Edge


class MemoryOrchestrator:
    """Main coordination - thin facade that delegates to specialized components."""
    
    def __init__(self, config: MemoryConfig, openai_api_key: Optional[str] = None,
                 llm_provider: Optional[LLMProvider] = None,
                 embedding_provider: Optional[EmbeddingProvider] = None,
                 store: Optional[Store] = None):
        self.config = config
        self.user_id = "default"
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        if llm_provider and embedding_provider:
            self.llm = llm_provider
            self.embedder = embedding_provider
        else:
            if not openai_api_key:
                raise ValueError("OpenAI API key required when not using custom providers")
            self.llm, self.embedder = create_resilient_providers(config, openai_api_key)
        
        # Initialize storage
        self.store = store or LanceDBStore(db_dir=config.db_dir)
        
        # Initialize core data structures
        self.shards: Dict[str, MemoryShard] = {}
        self.super_nodes: Dict[str, Node] = {}
        self.buffer = BufferGraph(self.shards, self.super_nodes)
        self.profile = Profile()
        
        # Initialize cache
        self.query_cache = QueryCache(max_size=config.cache_size) if config.enable_caching else None
        
        # Initialize specialized components
        self.retriever = MemoryRetriever(
            config=config,
            embedder=self.embedder,
            store=self.store,
            buffer=self.buffer,
            shards=self.shards,
            super_nodes=self.super_nodes,
            query_cache=self.query_cache
        )
        
        self.consolidator = MemoryConsolidator(
            config=config,
            llm=self.llm,
            embedder=self.embedder,
            store=self.store,
            buffer=self.buffer,
            shards=self.shards,
            super_nodes=self.super_nodes,
            profile=self.profile
        )
        
        self.profile_manager = ProfileManager(
            config=config,
            llm=self.llm,
            profile=self.profile
        )
        
        # Conversation state
        self.conversation_active = False
        self.short_term_memory: List[Dict] = []
        self.conversation_history: List[Dict] = []
        self.conversation_count = 0
        
        # Load from disk if configured
        if config.load_from_disk:
            self._load_from_persistence()
    
    def start_conversation(self) -> str:
        """Initialize a new interaction session."""
        self.conversation_active = True
        self.short_term_memory = []
        self.conversation_history = []
        return "✓ Conversation started"
    
    def chat(self, user_message: str) -> str:
        """Process a user message with memory retrieval and return LLM response."""
        if not self.conversation_active:
            self.start_conversation()
        
        start_time = time.time()
        
        # Add to short-term memory
        self.add_to_short_term(user_message, "episodic", salience=0.7)
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Retrieve relevant memories
        query_emb = self.retriever.get_embedding(user_message)
        retrieved_ids = self.retriever.optimized_retrieval(query_emb, user_message, self.user_id)
        
        # Boost neighbors
        self.retriever.boost_neighbors(retrieved_ids)
        
        retrieval_time = (time.time() - start_time) * 1000
        
        # Build context
        context_parts = []
        profile_context = self.profile_manager.get_context()
        if profile_context and profile_context != "No profile data yet.":
            context_parts.append(f"User Profile:\n{profile_context}\n")
        
        if retrieved_ids:
            memory_texts = []
            for nid in retrieved_ids:
                node = self.buffer.get_node(nid)
                if node:
                    memory_texts.append(f"- {node.content}")
                    self.buffer.update_access(nid)
            if memory_texts:
                context_parts.append(
                    "Relevant Information from Past Conversations (Use if relevant to the query):\n"
                    + "\n".join(memory_texts)
                    + "\n"
                )
        
        # Generate LLM response
        system_prompt = "You are a helpful assistant with access to the user's profile and past memories. Use the provided context ONLY if it is relevant to the user's current query. Do not force the information if it doesn't fit naturally."
        messages = [{"role": "system", "content": system_prompt}]
        
        if context_parts:
            messages.append({"role": "system", "content": "\n".join(context_parts)})
        
        for msg in self.conversation_history[-10:]:
            messages.append(msg)
        
        response = self.llm.completion(messages)
        self.add_to_short_term(response, "semantic", salience=0.5)
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Log performance
        timing_emoji = (
            "⚡" if retrieval_time < 100 else ("✓" if retrieval_time < 200 else "⏱")
        )
        self.logger.info(
            f"[{timing_emoji} Retrieval: {retrieval_time:.0f}ms, Retrieved: {len(retrieved_ids)} nodes]"
        )
        
        return response
    
    def chat_stream(self, user_message: str):
        """Stream LLM response while managing memory retrieval."""
        if not self.conversation_active:
            self.start_conversation()
        
        start_time = time.time()
        self.add_to_short_term(user_message, "episodic", salience=0.7)
        self.conversation_history.append({"role": "user", "content": user_message})
        
        query_emb = self.retriever.get_embedding(user_message)
        retrieved_ids = self.retriever.optimized_retrieval(query_emb, user_message, self.user_id)
        self.retriever.boost_neighbors(retrieved_ids)
        
        retrieval_time = (time.time() - start_time) * 1000
        
        # Yield metrics first
        timing_emoji = (
            "⚡" if retrieval_time < 100 else ("✓" if retrieval_time < 200 else "⏱")
        )
        yield {
            "type": "info",
            "content": f"[{timing_emoji} Retrieval: {retrieval_time:.0f}ms, Retrieved: {len(retrieved_ids)} nodes]",
        }
        
        if retrieved_ids:
            yield {"type": "info", "content": "   Retrieved Nodes:"}
            for nid in retrieved_ids:
                node = self.buffer.get_node(nid)
                if node:
                    snippet = (
                        node.content[:60] + "..."
                        if len(node.content) > 60
                        else node.content
                    )
                    yield {
                        "type": "info",
                        "content": f"   • [{nid}] ({node.shard_key}) {snippet}",
                    }
        
        # Build context
        context_parts = []
        profile_context = self.profile_manager.get_context()
        if profile_context and profile_context != "No profile data yet.":
            context_parts.append(f"User Profile:\n{profile_context}\n")
        
        if retrieved_ids:
            memory_texts = []
            for nid in retrieved_ids:
                node = self.buffer.get_node(nid)
                if node:
                    memory_texts.append(f"- {node.content}")
                    self.buffer.update_access(nid)
            if memory_texts:
                context_parts.append(
                    f"Relevant Information from Past Conversations (Use if relevant to the query):\n"
                    + "\n".join(memory_texts)
                    + "\n"
                )
        
        system_prompt = "You are a helpful assistant with access to the user's profile and past memories. Use the provided context ONLY if it is relevant to the user's current query. Do not force the information if it doesn't fit naturally."
        messages = [{"role": "system", "content": system_prompt}]
        
        if context_parts:
            messages.append({"role": "system", "content": "\n".join(context_parts)})
        
        for msg in self.conversation_history[-10:]:
            messages.append(msg)
        
        # Stream from LLM
        if hasattr(self.llm, "completion_stream"):
            full_response = ""
            stream = self.llm.completion_stream(messages)
            if stream:
                for chunk in stream:
                    full_response += chunk
                    yield {"type": "token", "content": chunk}
            
            # Post-stream cleanup
            self.add_to_short_term(full_response, "semantic", salience=0.5)
            self.conversation_history.append(
                {"role": "assistant", "content": full_response}
            )
        else:
            # Fallback to non-streaming
            response = self.llm.completion(messages)
            self.add_to_short_term(response, "semantic", salience=0.5)
            self.conversation_history.append({"role": "assistant", "content": response})
            yield {"type": "token", "content": response}
    
    def end_conversation(self) -> str:
        """Finalize the current session and trigger memory consolidation."""
        if not self.conversation_active:
            return "⚠ No active conversation to end."
        
        self.conversation_active = False
        if not self.short_term_memory:
            return "✓ Conversation ended. No memories to consolidate."
        
        results = []
        
        # Queue consolidation
        if self.consolidator.queue_consolidation(self.short_term_memory):
            results.append("✓ Conversation ended (consolidation queued)")
        else:
            results.append("✓ Conversation ended (consolidation completed)")
        
        # Apply temporal decay
        self.buffer.apply_temporal_decay(decay_rate=self.config.decay_rate)
        results.append("✓ Applied temporal decay")
        
        # Auto-pruning
        pruned = self.buffer.prune_weak_edges(threshold=self.config.prune_threshold)
        if pruned > 0:
            results.append(f"✓ Auto-pruned {pruned} weak edges")
        
        # Enforce buffer limit
        self._enforce_buffer_limit()
        
        self.conversation_count += 1
        
        # Auto-consolidation
        if self.config.auto_consolidate and self.conversation_count % self.config.consolidate_every == 0:
            self.logger.info(f"Auto-consolidation triggered (every {self.config.consolidate_every} conversations)...")
            consolidation_result = self.run_consolidation()
            results.append(consolidation_result)
        
        # Clear conversation state
        self.short_term_memory = []
        self.conversation_history = []
        
        # Save to persistence
        self._save_to_persistence()
        
        return "\n".join(results)
    
    def add_to_short_term(self, content: str, memory_type: str = "semantic", salience: float = 0.5):
        """Add a memory unit to the ephemeral buffer."""
        if not self.conversation_active:
            raise RuntimeError("No active conversation")
        
        memory = {
            "content": content,
            "type": memory_type,
            "salience": salience,
            "timestamp": time.time(),
        }
        self.short_term_memory.append(memory)
    
    def run_consolidation(self, weight_threshold: float = 0.6, merge_similar: bool = True) -> str:
        """Perform deep, graph-wide consolidation of memories."""
        results = []
        self.logger.info("🔄 Running consolidation...")
        
        if merge_similar:
            merged = self._merge_similar_nodes(similarity_threshold=0.95)
            if merged > 0:
                results.append(f"✓ Merged {merged} similar nodes")
        
        # Extract profile insights from connected components
        components = self.buffer.get_connected_components()
        profile_updates = 0
        
        for component in components:
            if len(component) < 3:
                continue
            
            # Calculate average edge weight
            component_edges = []
            for shard in self.shards.values():
                for (src, tgt), edge in shard.edges.items():
                    if src in component and tgt in component:
                        component_edges.append(edge)
            
            if not component_edges:
                continue
            
            avg_weight = sum(e.weight for e in component_edges) / len(component_edges)
            
            if avg_weight > 0.3:
                profile_update = self.profile_manager.extract_profile_from_component(component, self.buffer)
                if "Updated" in profile_update:
                    profile_updates += 1
                    results.append(profile_update)
        
        if profile_updates > 0:
            results.append(f"✓ Updated {profile_updates} profile domains")
        else:
            # Fallback profile extraction
            all_contents = [
                node.content for node in self.buffer.nodes.values() if not node.is_super_node
            ]
            if len(all_contents) >= 3:
                profile_update = self.profile_manager.extract_profile_from_contents(all_contents)
                if "Updated" in profile_update:
                    results.append(profile_update)
        
        if not results:
            results.append("✓ No consolidation actions needed")
        
        return "\n".join(results)
    
    def search_memories(self, query: str, limit: int = 5) -> List[Node]:
        """Perform semantic search for memories."""
        query_emb = self.retriever.get_embedding(query)
        node_ids = self.store.search_nodes(query_emb, user_id=self.user_id, limit=limit)
        
        results = []
        for nid in node_ids:
            node = self.buffer.get_node(nid)
            if node:
                results.append(node)
        return results
    
    def get_insights(self) -> str:
        """Get comprehensive user insights."""
        return self.profile_manager.get_comprehensive_insights(self.buffer)
    
    def get_stats(self) -> Dict:
        """Get comprehensive system statistics."""
        nodes, edges = self.buffer.size()
        
        # Aggregate stats from all components
        retriever_stats = self.retriever.get_stats()
        consolidator_stats = self.consolidator.get_stats()
        profile_stats = self.profile_manager.get_stats()
        
        return {
            "buffer_nodes": nodes,
            "buffer_edges": edges,
            "num_shards": len(self.shards),
            "num_super_nodes": len(self.super_nodes),
            "short_term_memories": len(self.short_term_memory),
            "conversation_active": self.conversation_active,
            "conversation_count": self.conversation_count,
            "user_id": self.user_id,
            "vector_store": "LanceDB (Active)",
            "performance": {
                **retriever_stats,
                **consolidator_stats,
                **profile_stats
            },
            "config": {
                "auto_consolidate": self.config.auto_consolidate,
                "consolidate_every": self.config.consolidate_every,
                "max_buffer_size": self.config.max_buffer_size,
                "enable_sharding": self.config.enable_sharding,
                "enable_hierarchy": self.config.enable_hierarchy,
                "enable_caching": self.config.enable_caching,
                "enable_async": self.config.enable_async
            }
        }
    
    def _enforce_buffer_limit(self):
        """Enforce maximum buffer size by removing old nodes."""
        nodes, _ = self.buffer.size()
        
        if nodes > self.config.max_buffer_size:
            excess = nodes - self.config.max_buffer_size
            all_nodes = []
            for shard in self.shards.values():
                for node_id, node in shard.nodes.items():
                    if not node.is_super_node:
                        days_old = (time.time() - node.last_accessed) / 86400
                        importance = (
                            node.salience * 0.5
                            + min(1.0, node.access_count / 10) * 0.3
                            + (1.0 / (1.0 + days_old)) * 0.2
                        )
                        all_nodes.append((node_id, importance, node.shard_key))
            
            all_nodes.sort(key=lambda x: x[1])
            to_remove = all_nodes[:excess]
            
            removed_count = 0
            for node_id, _, shard_key in to_remove:
                if shard_key in self.shards:
                    shard = self.shards[shard_key]
                    if node_id in shard.nodes:
                        del shard.nodes[node_id]
                        removed_count += 1
                        
                        edges_to_remove = [
                            k for k, e in shard.edges.items()
                            if e.source == node_id or e.target == node_id
                        ]
                        for key in edges_to_remove:
                            del shard.edges[key]
            
            if removed_count > 0:
                # Sync with LanceDB
                to_remove_ids = [nid for nid, _, _ in to_remove]
                self.store.delete_nodes(to_remove_ids, user_id=self.user_id)
                
                self.logger.warning(
                    f"Buffer limit reached! Archived {removed_count} old nodes (limit: {self.config.max_buffer_size})"
                )
    
    def _merge_similar_nodes(self, similarity_threshold: float = 0.95) -> int:
        """Merge near-identical nodes to reduce redundancy."""
        if len(self.buffer.nodes) < 2:
            return 0
        
        merged_count = 0
        processed = set()
        node_list = list(self.buffer.nodes.items())
        
        for i, (nid1, node1) in enumerate(node_list):
            if nid1 in processed or node1.is_super_node:
                continue
            
            for j in range(i + 1, len(node_list)):
                nid2, node2 = node_list[j]
                if nid2 in processed or node2.is_super_node:
                    continue
                
                sim = self.retriever._cosine_similarity(node1.embedding, node2.embedding)
                if sim > similarity_threshold:
                    node1.content = f"{node1.content} | {node2.content}"
                    node1.salience = max(node1.salience, node2.salience)
                    node1.access_count += node2.access_count
                    
                    # Update edges
                    for shard in self.shards.values():
                        if nid2 in shard.nodes:
                            edges_to_update = []
                            for (src, tgt), edge in shard.edges.items():
                                if src == nid2:
                                    edges_to_update.append(((src, tgt), (nid1, tgt)))
                                elif tgt == nid2:
                                    edges_to_update.append(((src, tgt), (src, nid1)))
                            
                            for old_key, new_key in edges_to_update:
                                edge = shard.edges[old_key]
                                del shard.edges[old_key]
                                edge.source, edge.target = new_key
                                shard.edges[new_key] = edge
                            
                            del shard.nodes[nid2]
                            break
                    
                    processed.add(nid2)
                    merged_count += 1
                    
                    # Sync with LanceDB
                    self.store.delete_nodes([nid2], user_id=self.user_id)
                    self.store.add_nodes([{
                        "id": nid1,
                        "content": node1.content,
                        "embedding": node1.embedding,
                        "type": node1.type,
                        "salience": node1.salience,
                        "shard_key": node1.shard_key,
                        "timestamp": node1.timestamp
                    }], user_id=self.user_id)
        
        return merged_count
    
    def _save_to_persistence(self):
        """Save current state to LanceDB Store."""
        # Save nodes
        all_nodes = []
        for node in self.buffer.nodes.values():
            all_nodes.append(node.to_dict())
        
        if all_nodes:
            self.store.delete_nodes([], user_id=self.user_id)  # Clear existing
            self.store.add_nodes(all_nodes, user_id=self.user_id)
        
        # Save edges
        all_edges = []
        for shard in self.shards.values():
            for edge in shard.edges.values():
                all_edges.append(edge.to_dict())
        
        if all_edges:
            self.store.delete_edges(user_id=self.user_id)  # Clear existing
            self.store.add_edges(all_edges, user_id=self.user_id)
        
        # Save profile
        self.store.save_profile(self.profile.to_dict(), user_id=self.user_id)
        
        self.logger.info(f"State persisted to LanceDB for user: {self.user_id}")
    
    def _load_from_persistence(self):
        """Load state from LanceDB Store."""
        self.logger.info(f"Loading state from LanceDB for user: {self.user_id}...")
        
        # Load nodes
        node_dicts = self.store.get_nodes(user_id=self.user_id)
        if not node_dicts:
            self.logger.info("No saved state found in LanceDB.")
            return
        
        self.shards = {}
        self.super_nodes = {}
        
        for nd in node_dicts:
            # Map database field 'vector' back to 'embedding'
            if "vector" in nd:
                nd["embedding"] = nd.pop("vector")
            
            node = Node.from_dict(nd)
            
            if node.is_super_node:
                self.super_nodes[node.id] = node
            else:
                s_key = node.shard_key
                if s_key not in self.shards:
                    self.shards[s_key] = MemoryShard(s_key)
                self.shards[s_key].add_node(node)
        
        # Load edges
        edge_dicts = self.store.get_edges(user_id=self.user_id)
        for ed in edge_dicts:
            # Map database fields back to Edge format
            if "source_id" in ed:
                ed["source"] = ed.pop("source_id")
            if "target_id" in ed:
                ed["target"] = ed.pop("target_id")
            
            edge = Edge.from_dict(ed)
            
            # Find which shard this edge belongs to
            src_node = None
            if edge.source in self.super_nodes:
                src_node = self.super_nodes[edge.source]
            else:
                for shard in self.shards.values():
                    if edge.source in shard.nodes:
                        src_node = shard.nodes[edge.source]
                        break
            
            if src_node:
                s_key = src_node.shard_key
                if s_key in self.shards:
                    self.shards[s_key].add_edge(edge)
        
        # Load profile
        profile_data = self.store.load_profile(user_id=self.user_id)
        if profile_data:
            self.profile = Profile.from_dict(profile_data)
        
        # Re-initialize buffer
        self.buffer = BufferGraph(self.shards, self.super_nodes)
        
        # Update node counter
        if self.buffer.nodes:
            max_id = 0
            for nid in self.buffer.nodes.keys():
                if nid.startswith("node_"):
                    try:
                        num = int(nid.split("_")[1])
                        max_id = max(max_id, num)
                    except:
                        pass
            self.consolidator.node_counter = max_id
        
        self.logger.info(f"Restored state from LanceDB ({len(self.buffer.nodes)} nodes, {len(edge_dicts)} edges)")
    
    def switch_user(self, new_user_id: str):
        """Switch memory system context to a different user."""
        if self.conversation_active:
            self.end_conversation()
        else:
            self._save_to_persistence()
        
        self.user_id = new_user_id
        self._load_from_persistence()
        self.logger.info(f"Switched context to user: {new_user_id}")
    
    def close(self):
        """Close the memory system and its resources."""
        self._save_to_persistence()
        self.consolidator.close()
        if hasattr(self.store, 'close'):
            self.store.close()