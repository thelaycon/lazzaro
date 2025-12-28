import time
import logging
import json
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from ..models.graph import Node, Edge
from .buffer_graph import BufferGraph
from .interfaces import LLMProvider, EmbeddingProvider, Store
from .memory_shard import MemoryShard
from .profile import Profile


class MemoryConsolidator:
    """Focused on background processing and memory consolidation."""
    
    def __init__(self, config: 'MemoryConfig', llm: LLMProvider, embedder: EmbeddingProvider,
                 store: Store, buffer: BufferGraph, shards: Dict[str, MemoryShard],
                 super_nodes: Dict[str, Node], profile: Profile):
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.buffer = buffer
        self.shards = shards
        self.super_nodes = super_nodes
        self.profile = profile
        self.logger = logging.getLogger(__name__)
        
        # Background processing
        self.consolidation_queue: List[Dict] = []
        self.background_executor = (
            ThreadPoolExecutor(max_workers=2) if config.enable_async else None
        )
        self.node_counter = 0
        self.metrics = {"consolidation_times": [], "llm_calls": 0}
    
    def queue_consolidation(self, memories: List[Dict]):
        """Queue memories for background consolidation."""
        if self.config.enable_async and self.background_executor:
            consolidation_data = {
                "memories": memories.copy(),
                "timestamp": time.time(),
            }
            self.consolidation_queue.append(consolidation_data)
            self.background_executor.submit(self._async_consolidate)
            return True
        else:
            # Synchronous consolidation
            return self._consolidate_memories(memories)
    
    def _async_consolidate(self):
        """Background consolidation worker."""
        if not self.consolidation_queue:
            return
        
        start_time = time.time()
        all_memories = []
        for batch in self.consolidation_queue:
            all_memories.extend(batch["memories"])
        
        self.consolidation_queue.clear()
        self.logger.info(f"Processing {len(all_memories)} memories in background...")
        
        self._consolidate_memories(all_memories)
        
        elapsed = time.time() - start_time
        self.metrics["consolidation_times"].append(elapsed)
        self.logger.info(f"Background consolidation complete ({elapsed:.2f}s)")
    
    def _consolidate_memories(self, memories: List[Dict]) -> bool:
        """Consolidate a list of memories into the graph."""
        if not memories:
            return False
        
        # Extract facts using LLM
        extracted_memories = self._extract_facts(memories)
        if not extracted_memories:
            return False
        
        # Create nodes and embeddings
        new_nodes = self._create_memory_nodes(extracted_memories)
        if not new_nodes:
            return False
        
        # Link memories
        self._link_within_shards(new_nodes)
        self._link_to_existing_memories(new_nodes)
        
        # Create super nodes if needed
        if self.config.enable_hierarchy:
            self._update_super_nodes()
        
        # Apply temporal decay and pruning
        self.buffer.apply_temporal_decay(decay_rate=self.config.decay_rate)
        self.buffer.prune_weak_edges(threshold=self.config.prune_threshold)
        
        return True
    
    def _extract_facts(self, memories: List[Dict]) -> List[Dict]:
        """Extract atomic facts from conversation memories."""
        conv_text = json.dumps(memories)
        system_prompt = """Extract distinct, atomic facts from this conversation.
Categorization Guidelines:
1. semantic: Stable facts, preferences, or knowledge (e.g., "User likes Python", "User lives in London").
2. episodic: Specific events, occurrences, or recent activities (e.g., "User started a new job today", "User fixed a bug in the API").
3. procedural: Processes, workflows, or instructions (e.g., "User follows the git-flow model", "User prefers TDD for testing").

Format Rules:
- Formulate facts in the THIRD PERSON.
- Abstract from conversational filler.
- If no new facts, return empty list.

Return JSON: {"memories": [{"content": "...", "type": "semantic|episodic|procedural", "salience": 0.0-1.0, "topic": "work|personal|learning|health|other"}]}
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conv_text},
        ]
        
        try:
            self.metrics["llm_calls"] += 1
            response = self.llm.completion(messages, response_format={"type": "json_object"})
            
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            
            data = json.loads(response)
            
            # Handle both {"memories": [...]} and [...] formats
            if isinstance(data, dict):
                memories = data.get("memories", [])
            elif isinstance(data, list):
                memories = data
            else:
                self.logger.warning(f"Unexpected data type: {type(data)}")
                return []
            
            # Sanitize: ensure all memories are dictionaries
            return [m for m in memories if isinstance(m, dict)]
            
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Fact extraction failed: {e}")
            return []
    
    def _create_memory_nodes(self, memories: List[Dict]) -> List[Tuple[str, str]]:
        """Create memory nodes from extracted facts."""
        contents = [m.get("content", "") for m in memories if m.get("content")]
        if not contents:
            return []
        
        # Batch embed for efficiency
        embeddings = self.embedder.batch_embed(contents)
        
        new_nodes = []
        new_nodes_data = []
        
        for i, mem in enumerate(memories):
            content = mem.get("content", "")
            if not content or len(content) < 5:
                continue
            
            shard_key = mem.get("topic", self._infer_shard_key(content))
            shard = self._get_or_create_shard(shard_key)
            
            # Check for existing duplicate via Vector Store
            new_emb = embeddings[i] if i < len(embeddings) else []
            existing_node = None
            
            if new_emb:
                results = self.store.search_nodes(new_emb, user_id="default", limit=1)
                if results:
                    best_match_id = results[0]
                    best_node = self.buffer.get_node(best_match_id)
                    if best_node:
                        sim = self._cosine_similarity(new_emb, best_node.embedding)
                        if sim > 0.95:
                            existing_node = best_node
            
            if existing_node:
                existing_node.salience = max(existing_node.salience, mem.get("salience", 0.5))
                existing_node.last_accessed = time.time()
                existing_node.access_count += 1
                self.logger.info(f"Merged semantic duplicate into {existing_node.id}")
                continue
            
            # Create new node
            node_id = self._generate_node_id()
            node = Node(
                id=node_id,
                content=content,
                embedding=new_emb,
                type=mem.get("type", "semantic"),
                salience=mem.get("salience", 0.5),
                shard_key=shard_key,
            )
            
            shard.add_node(node)
            new_nodes.append((node_id, shard_key))
            
            # Prepare for Vector Store insertion
            new_nodes_data.append({
                "id": node_id,
                "content": content,
                "embedding": new_emb,
                "type": node.type,
                "salience": node.salience,
                "shard_key": node.shard_key,
                "timestamp": node.timestamp
            })
        
        # Add to vector store
        if new_nodes_data:
            self.store.add_nodes(new_nodes_data, user_id="default")
        
        return new_nodes
    
    def _link_within_shards(self, new_nodes: List[Tuple[str, str]]):
        """Create links between new nodes within the same shard."""
        shard_groups = defaultdict(list)
        for node_id, shard_key in new_nodes:
            shard_groups[shard_key].append(node_id)
        
        for shard_key, node_ids in shard_groups.items():
            if len(node_ids) < 2:
                continue
            
            shard = self.shards[shard_key]
            
            # Sequential linking
            for i in range(len(node_ids) - 1):
                edge = Edge(
                    source=node_ids[i],
                    target=node_ids[i + 1],
                    weight=0.5,
                    edge_type="relates_to",
                )
                shard.add_edge(edge)
            
            # Semantic similarity linking
            for node_id in node_ids:
                node = shard.nodes[node_id]
                similarities = []
                for existing_id, existing_node in shard.nodes.items():
                    if existing_id == node_id or existing_id in node_ids:
                        continue
                    sim = self._cosine_similarity(node.embedding, existing_node.embedding)
                    similarities.append((existing_id, sim))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                for existing_id, sim in similarities[:3]:
                    if sim > 0.5:
                        edge = Edge(
                            source=node_id,
                            target=existing_id,
                            weight=sim * 0.8,
                            edge_type="relates_to",
                        )
                        shard.add_edge(edge)
    
    def _link_to_existing_memories(self, new_nodes: List[Tuple[str, str]]):
        """Create links between new nodes and existing memories."""
        if not new_nodes:
            return
        
        new_ids = {nid for nid, _ in new_nodes}
        existing_nodes = {}
        for shard in self.shards.values():
            for node_id, node in shard.nodes.items():
                if node_id not in new_ids and not node.is_super_node:
                    existing_nodes[node_id] = node
        
        if not existing_nodes:
            return
        
        links_created = 0
        for new_id, new_shard_key in new_nodes:
            new_node = self.buffer.get_node(new_id)
            if not new_node:
                continue
            
            similarities = []
            for existing_id, existing_node in existing_nodes.items():
                sim = self._cosine_similarity(new_node.embedding, existing_node.embedding)
                similarities.append((existing_id, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            for existing_id, similarity in similarities[:3]:
                if similarity > 0.5:
                    existing_key_1 = (new_id, existing_id)
                    existing_key_2 = (existing_id, new_id)
                    edge_exists = False
                    for shard in self.shards.values():
                        if existing_key_1 in shard.edges or existing_key_2 in shard.edges:
                            edge_exists = True
                            break
                    
                    if not edge_exists:
                        edge = Edge(
                            source=new_id,
                            target=existing_id,
                            weight=similarity * 0.8,
                            edge_type="relates_to",
                        )
                        shard = self.shards.get(new_shard_key)
                        if shard:
                            shard.add_edge(edge)
                            links_created += 1
        
        if links_created > 0:
            self.logger.info(f"Created {links_created} cross-conversation links")
    
    def _update_super_nodes(self):
        """Update or create super nodes for dense shards."""
        for shard_key, shard in self.shards.items():
            if len(shard.nodes) > self.config.super_node_threshold:
                self._create_super_node_for_shard(shard_key)
    
    def _create_super_node_for_shard(self, shard_key: str):
        """Create a super node for a given shard."""
        shard = self.shards[shard_key]
        if len(shard.nodes) < self.config.super_node_threshold:
            return
        
        existing_super = [
            n for n in self.super_nodes.values() if n.shard_key == shard_key
        ]
        if existing_super:
            return
        
        self.logger.info(f"Creating super-node for shard '{shard_key}' ({len(shard.nodes)} nodes)")
        
        nodes = list(shard.nodes.values())
        super_id = f"super_{shard_key}_{int(time.time())}"
        
        sample_contents = [n.content for n in nodes[:10]]
        aggregated_content = (
            f"Topic: {shard_key}. Contains memories about: "
            + "; ".join(sample_contents[:3])
        )
        
        embeddings = [n.embedding for n in nodes if n.embedding]
        avg_embedding = np.mean(embeddings, axis=0).tolist() if embeddings else []
        
        super_node = Node(
            id=super_id,
            content=aggregated_content,
            embedding=avg_embedding,
            type="semantic",
            is_super_node=True,
            child_ids=[n.id for n in nodes],
            shard_key=shard_key,
        )
        
        for node in nodes:
            node.parent_id = super_id
        
        self.super_nodes[super_id] = super_node
        self.logger.info(f"Created super-node {super_id} with {len(nodes)} children")
    
    def _infer_shard_key(self, content: str) -> str:
        """Categorize content into a semantic shard based on keywords or date."""
        if not self.config.enable_sharding:
            return "default"
        
        keywords = {
            "work": ["work", "project", "meeting", "deadline", "client", "colleague"],
            "personal": ["family", "friend", "hobby", "home", "personal"],
            "learning": ["learn", "study", "course", "book", "tutorial", "practice"],
            "health": ["health", "exercise", "diet", "sleep", "medical", "fitness"],
        }
        
        content_lower = content.lower()
        for shard_key, terms in keywords.items():
            if any(term in content_lower for term in terms):
                return shard_key
        
        return time.strftime("%Y-%m")
    
    def _get_or_create_shard(self, shard_key: str) -> MemoryShard:
        """Retrieve an existing shard or create a new one."""
        if shard_key not in self.shards:
            self.shards[shard_key] = MemoryShard(shard_key)
        return self.shards[shard_key]
    
    def _generate_node_id(self) -> str:
        """Generate a unique ID for new memory nodes."""
        self.node_counter += 1
        return f"node_{self.node_counter}"
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate normalized cosine similarity between two vectors."""
        if not v1 or not v2:
            return 0.0
        try:
            import numpy as np
            a, b = np.array(v1), np.array(v2)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            return float(np.dot(a, b) / norm) if norm > 0 else 0.0
        except ImportError:
            # Fallback without numpy
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm_a = sum(a * a for a in v1) ** 0.5
            norm_b = sum(b * b for b in v2) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """Get consolidation performance statistics."""
        avg_consolidation = (
            np.mean(self.metrics["consolidation_times"])
            if self.metrics["consolidation_times"]
            else 0
        )
        
        return {
            "avg_consolidation_s": f"{avg_consolidation:.2f}",
            "total_consolidations": len(self.metrics["consolidation_times"]),
            "llm_calls": self.metrics["llm_calls"],
            "queue_size": len(self.consolidation_queue)
        }
    
    def close(self):
        """Clean up resources."""
        if self.background_executor:
            self.background_executor.shutdown(wait=True)