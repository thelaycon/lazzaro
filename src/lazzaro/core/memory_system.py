import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import openai

from ..models.graph import Edge, Node
from .buffer_graph import BufferGraph
from .interfaces import EmbeddingProvider, LLMProvider
from .memory_shard import MemoryShard
from .persistence import PersistenceManager
from .profile import Profile
from .providers import OpenAIEmbedder, OpenAILLM
from .query_cache import QueryCache


class MemorySystem:
    """
    The orchestrator for the Lazzaro Scalable Memory System.

    MemorySystem manages the entire lifecycle of an AI agent's memory, including
    short-term buffering, long-term graph consolidation, hierarchical clustering,
    and semantic sharding.

    Args:
        openai_api_key: Optional OpenAI API key if using default providers.
        model: LLM model name (default: "gpt-4o-mini").
        enable_sharding: If True, organize memories into semantic shards (subgraphs).
        enable_hierarchy: If True, cluster dense shards into super-nodes for faster search.
        enable_caching: If True, cache LLM and embedding calls.
        enable_async: If True, run consolidation in background threads.
        max_shard_size: Maximum nodes per shard before considering splits.
        super_node_threshold: Number of nodes in a shard before creating a super-node.
        auto_consolidate: If True, trigger consolidation after N conversations.
        consolidate_every: Number of conversations between auto-consolidations.
        auto_prune: If True, automatically prune weak associations.
        prune_threshold: Minimum edge weight to retain.
        max_buffer_size: Maximum nodes in the graph before archiving old ones.
        load_from_disk: If True, reload state from 'db/lazzaro.pkl' on init.
        llm_provider: Custom LLM implementation (must follow LLMProvider protocol).
        embedding_provider: Custom Embedding implementation (must follow EmbeddingProvider protocol).

    Example:
        ```python
        from lazzaro.core.memory_system import MemorySystem
        from lazzaro.core.providers import GeminiLLM, GeminiEmbedder

        llm = GeminiLLM(api_key="...", model="gemini-1.5-flash")
        embedder = GeminiEmbedder(api_key="...")

        ms = MemorySystem(llm_provider=llm, embedding_provider=embedder)

        ms.start_conversation()
        response = ms.chat("My favorite color is blue.")
        ms.end_conversation()
        ```
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        enable_sharding: bool = True,
        enable_hierarchy: bool = True,
        enable_caching: bool = True,
        enable_async: bool = True,
        max_shard_size: int = 500,
        super_node_threshold: int = 20,
        auto_consolidate: bool = True,
        consolidate_every: int = 3,
        auto_prune: bool = True,
        prune_threshold: float = 0.5,
        max_buffer_size: int = 10,
        load_from_disk: bool = True,
        llm_provider: Optional[LLMProvider] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.model = model

        # Initialize providers
        if llm_provider:
            self.llm = llm_provider
        else:
            self.llm = OpenAILLM(api_key=openai_api_key, model=model)

        if embedding_provider:
            self.embedder = embedding_provider
        else:
            self.embedder = OpenAIEmbedder(api_key=openai_api_key)

        self.shards: Dict[str, MemoryShard] = {}
        self.super_nodes: Dict[str, Node] = {}
        self.buffer = BufferGraph(self.shards, self.super_nodes)
        self.profile = Profile()
        self.persistence = PersistenceManager()

        self.enable_sharding = enable_sharding
        self.enable_hierarchy = enable_hierarchy
        self.enable_caching = enable_caching
        self.enable_async = enable_async
        self.max_shard_size = max_shard_size
        self.super_node_threshold = super_node_threshold

        self.auto_consolidate = auto_consolidate
        self.consolidate_every = consolidate_every
        self.auto_prune = auto_prune
        self.prune_threshold = prune_threshold
        self.max_buffer_size = max_buffer_size

        self.query_cache = QueryCache(max_size=1000) if enable_caching else None
        self.consolidation_queue: List[Dict] = []
        self.background_executor = (
            ThreadPoolExecutor(max_workers=2) if enable_async else None
        )

        self.conversation_active = False
        self.short_term_memory: List[Dict] = []
        self.conversation_history: List[Dict] = []
        self.node_counter = 0
        self.conversation_count = 0

        self.metrics = {
            "embedding_calls": 0,
            "llm_calls": 0,
            "retrieval_times": [],
            "consolidation_times": [],
        }

        if load_from_disk:
            self._load_from_persistence()

    def _generate_node_id(self) -> str:
        """Generates a unique ID for new memory nodes."""
        self.node_counter += 1
        return f"node_{self.node_counter}"

    def _infer_shard_key(self, content: str) -> str:
        """Categorizes content into a semantic shard based on keywords or date."""
        if not self.enable_sharding:
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
        """Retrieves an existing shard or creates a new one for the given key."""
        if shard_key not in self.shards:
            self.shards[shard_key] = MemoryShard(shard_key)
        return self.shards[shard_key]

    def _get_embedding(self, text: str) -> List[float]:
        """Fetches vector embedding for text, using cache if enabled."""
        self.metrics["embedding_calls"] += 1
        if self.query_cache:
            cached = self.query_cache.get_embedding(text)
            if cached:
                return cached

        embedding = self.embedder.embed(text)
        if self.query_cache:
            self.query_cache.set_embedding(text, embedding)
        return embedding

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Handles batch embedding requests to minimize provider overhead."""
        if not texts:
            return []
        self.metrics["embedding_calls"] += 1
        return self.embedder.batch_embed(texts)

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculates normalized cosine similarity between two vectors."""
        if not v1 or not v2:
            return 0.0
        a, b = np.array(v1), np.array(v2)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / norm) if norm > 0 else 0.0

    def _call_llm(self, messages: List[Dict], response_format: Dict = None) -> str:
        """Calls the LLM provider and tracks metrics."""
        self.metrics["llm_calls"] += 1
        return self.llm.completion(messages, response_format)

    def start_conversation(self) -> str:
        """
        Initializes a new interaction session.
        Clears ephemeral conversation history and short-term buffer.
        """
        self.conversation_active = True
        self.short_term_memory = []
        self.conversation_history = []
        return "✓ Conversation started"

    def add_to_short_term(
        self, content: str, memory_type: str = "semantic", salience: float = 0.5
    ):
        """
        Manually adds a memory unit to the ephemeral buffer.
        This buffer is processed during consolidation at the end of the conversation.
        """
        if not self.conversation_active:
            raise RuntimeError("No active conversation")
        memory = {
            "content": content,
            "type": memory_type,
            "salience": salience,
            "timestamp": time.time(),
        }
        self.short_term_memory.append(memory)
        self._auto_save_if_needed()

    def _auto_save_if_needed(self):
        # Save short term changes occasionally if critical
        pass  # Currently saving primarily on end/consolidation for efficiency

    def _boost_neighbors(self, retrieved_ids: List[str]):
        """Latency-Aware Boosting: Pull neighbors into the present"""
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
            print(f"   (Graph: Boosted {count} neighbor nodes via association)")

    def chat(self, user_message: str) -> str:
        """
        Processes a user message, retrieves relevant memories, and returns an LLM response.

        This is the primary synchronous entry point for interactions. It performs:
        1. Contextual memory retrieval (semantic + hierarchical).
        2. Associative neighbor boosting.
        3. LLM completion with injected memory context.

        Args:
            user_message: The raw text input from the user.

        Returns:
            The assistant's text response.

        Example:
            ```python
            ms = MemorySystem(...)
            response = ms.chat("What did we discuss about the project yesterday?")
            print(response)
            ```
        """
        if not self.conversation_active:
            print(self.start_conversation())

        start_time = time.time()
        self.add_to_short_term(user_message, "episodic", salience=0.7)
        self.conversation_history.append({"role": "user", "content": user_message})

        query_emb = self._get_embedding(user_message)
        retrieved_ids = self._optimized_retrieval(query_emb, user_message)

        # Latency-Aware Boosting (Neighbors inherit timestamp)
        self._boost_neighbors(retrieved_ids)

        retrieval_time = (time.time() - start_time) * 1000
        self.metrics["retrieval_times"].append(retrieval_time)

        context_parts = []
        profile_context = self.profile.get_context()
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
                    "Relevant Information from Past Conversations (You MUST use this if relevant):\n"
                    + "\n".join(memory_texts)
                    + "\n"
                )

        system_prompt = "You are a helpful assistant with access to the user's profile and past memories. These memories are factual records of the user's life and preferences. You MUST use this context to answer questions about the user."
        messages = [{"role": "system", "content": system_prompt}]

        if context_parts:
            messages.append({"role": "system", "content": "\n".join(context_parts)})

        for msg in self.conversation_history[-10:]:
            messages.append(msg)

        response = self._call_llm(messages)
        self.add_to_short_term(response, "semantic", salience=0.5)
        self.conversation_history.append({"role": "assistant", "content": response})

        timing_emoji = (
            "⚡" if retrieval_time < 100 else ("✓" if retrieval_time < 200 else "⏱")
        )
        print(
            f"[{timing_emoji} Retrieval: {retrieval_time:.0f}ms, Retrieved: {len(retrieved_ids)} nodes]"
        )

        # List retrieved nodes
        if retrieved_ids:
            print("   Retrieved Nodes:")
            for nid in retrieved_ids:
                node = self.buffer.get_node(nid) # Fix: ensure node is defined
                if node:
                    snippet = (
                        node.content[:60] + "..."
                        if len(node.content) > 60
                        else node.content
                    )
                    print(f"   • [{nid}] ({node.shard_key}) {snippet}")
        return response

    def chat_stream(self, user_message: str):
        """
        Streams the LLM response chunk by chunk while managing memory retrieval.

        Yields:
            Dictionaries with 'type' ("info" or "token") and 'content'.

        Example:
            ```python
            for chunk in ms.chat_stream("Explain the project again"):
                if chunk['type'] == 'token':
                    print(chunk['content'], end="", flush=True)
            ```
        """
        if not self.conversation_active:
            print(self.start_conversation())

        start_time = time.time()
        self.add_to_short_term(user_message, "episodic", salience=0.7)
        self.conversation_history.append({"role": "user", "content": user_message})

        query_emb = self._get_embedding(user_message)
        retrieved_ids = self._optimized_retrieval(query_emb, user_message)
        self._boost_neighbors(retrieved_ids)

        retrieval_time = (time.time() - start_time) * 1000

        # 1. Yield metrics/info first
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

        # 2. Build context
        context_parts = []
        profile_context = self.profile.get_context()
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
                    f"Relevant Information from Past Conversations (You MUST use this):\n"
                    + "\n".join(memory_texts)
                    + "\n"
                )

        system_prompt = "You are a helpful assistant with access to the user's profile and past memories. These memories are factual records of the user's life and preferences. You MUST use this context to answer questions about the user."
        messages = [{"role": "system", "content": system_prompt}]

        if context_parts:
            messages.append({"role": "system", "content": "\n".join(context_parts)})

        for msg in self.conversation_history[-10:]:
            messages.append(msg)

        # 3. Stream from LLM provider
        # Fallback if provider doesn't support streaming (though Protocol says it should)
        if hasattr(self.llm, "completion_stream"):
            full_response = ""
            for chunk in self.llm.completion_stream(messages):
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

    def _optimized_retrieval(
        self, query_emb: List[float], query_text: str
    ) -> List[str]:
        if self.query_cache:
            cached = self.query_cache.get_results(query_text)
            if cached:
                return cached

        retrieved = []

        if self.enable_hierarchy and self.super_nodes:
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

        relevant_shards = self._get_relevant_shards(query_text)
        all_scores = []
        for shard_key in relevant_shards:
            if shard_key not in self.shards:
                continue

            shard = self.shards[shard_key]
            shard.last_accessed = time.time()
            shard.access_count += 1

            for node_id, node in shard.nodes.items():
                if node.is_super_node:
                    continue
                sim = self._cosine_similarity(query_emb, node.embedding)
                days_old = (time.time() - node.last_accessed) / 86400
                recency_factor = 0.95**days_old
                score = sim * recency_factor
                all_scores.append((node_id, score))

        all_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter duplicates and select top 5
        seen_content = set()
        for nid, score in all_scores:
            if score <= 0.25:
                continue

            if len(retrieved) >= 5:
                break

            node = self.buffer.get_node(nid)
            if node:
                # Deduplication: specific enough content shouldn't be repeated
                if node.content in seen_content:
                    continue
                seen_content.add(node.content)
                retrieved.append(nid)

        if self.query_cache:
            self.query_cache.set_results(query_text, retrieved)
        return retrieved

    def _get_relevant_shards(self, query: str, max_shards: int = 3) -> List[str]:
        if not self.enable_sharding or not self.shards:
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

    def _enforce_buffer_limit(self):
        nodes, _ = self.buffer.size()

        if nodes > self.max_buffer_size:
            excess = nodes - self.max_buffer_size
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
                            k
                            for k, e in shard.edges.items()
                            if e.source == node_id or e.target == node_id
                        ]
                        for key in edges_to_remove:
                            del shard.edges[key]

            if removed_count > 0:
                print(
                    f"⚠ Buffer limit reached! Archived {removed_count} old nodes (limit: {self.max_buffer_size})"
                )

    def end_conversation(self) -> str:
        """
        Finalizes the current session and triggers the memory consolidation pipeline.

        This method:
        1. Closes the active conversation.
        2. Queues recent interactions for background fact extraction (consolidation).
        3. Applies temporal decay to the entire graph.
        4. Triggers auto-pruning and auto-consolidation if thresholds are met.
        5. Persists the updated state to disk.

        Returns:
            A status message summarizing the operations performed.

        Example:
            ```python
            status = ms.end_conversation()
            print(status)
            ```
        """
        if not self.conversation_active:
            return "⚠ No active conversation to end."

        self.conversation_active = False
        if not self.short_term_memory:
            return "✓ Conversation ended. No memories to consolidate."

        results = []
        if self.enable_async and self.background_executor:
            print(
                f"🔄 Queueing consolidation for {len(self.short_term_memory)} exchanges..."
            )
            consolidation_data = {
                "memories": self.short_term_memory.copy(),
                "timestamp": time.time(),
            }
            self.consolidation_queue.append(consolidation_data)
            self.background_executor.submit(self._async_consolidate)
            results.append("✓ Conversation ended (consolidation queued)")
        else:
            print(f"🔄 Consolidating {len(self.short_term_memory)} exchanges...")
            result = self._consolidate_to_buffer()
            results.append(result)

        self.buffer.apply_temporal_decay(decay_rate=0.01)
        results.append("✓ Applied temporal decay")

        if self.auto_prune:
            pruned = self.buffer.prune_weak_edges(threshold=self.prune_threshold)
            if pruned > 0:
                results.append(f"✓ Auto-pruned {pruned} weak edges")

        self._enforce_buffer_limit()
        self.conversation_count += 1

        if (
            self.auto_consolidate
            and self.conversation_count % self.consolidate_every == 0
        ):
            print(
                f"🔄 Auto-consolidation triggered (every {self.consolidate_every} conversations)..."
            )
            consolidation_result = self.run_consolidation()
            results.append(consolidation_result)

        self.short_term_memory = []
        self.conversation_history = []

        self._save_to_persistence()
        return "\n".join(results)

    def _async_consolidate(self):
        if not self.consolidation_queue:
            return

        start_time = time.time()
        all_memories = []
        for batch in self.consolidation_queue:
            all_memories.extend(batch["memories"])

        self.consolidation_queue.clear()
        print(f"🔄 Processing {len(all_memories)} memories in background...")

        conv_text = json.dumps(all_memories)
        system_prompt = """Extract distinct, atomic facts from this conversation.
Guidelines:
1. Facts must be about the user or their preferences/experiences.
2. Formulate facts in the THIRD PERSON (e.g., 'The user likes...' not 'I like...').
3. Abstract the fact from conversational filler (e.g., 'User prefers Python' instead of 'The user said they prefer Python').
4. DO NOT return raw messages word-for-word.
5. If no new factual information is found, return an empty list of memories.

Return JSON: {"memories": [{"content": "...", "type": "semantic|episodic|procedural", "salience": 0.0-1.0, "topic": "work|personal|learning|health|other"}]}

Example:
Input: "I love football"
Output: {"memories": [{"content": "User loves football", "type": "episodic", "salience": 1.0, "topic": "personal"}]}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conv_text},
        ]
        response = self._call_llm(messages, response_format={"type": "json_object"})

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            data = json.loads(response)

            # Handle both {"memories": [...]} and [...] formats
            if isinstance(data, dict):
                memories = data.get("memories", [])
            elif isinstance(data, list):
                memories = data
            else:
                print(f"⚠ Unexpected data type: {type(data)}")
                return
        except json.JSONDecodeError as e:
            print(f"⚠ Parse error: {e}")
            return

        # Sanitize: ensure all memories are dictionaries
        if isinstance(memories, list):
            memories = [m for m in memories if isinstance(m, dict)]

        print(f"✓ Extracted {len(memories)} memory candidates")
        contents = [m.get("content", "") for m in memories if m.get("content")]
        embeddings = self._batch_embed(contents)

        new_nodes = []
        for i, mem in enumerate(memories):
            content = mem.get("content", "")
            if not content or len(content) < 5:
                continue

            shard_key = mem.get("topic", self._infer_shard_key(content))
            shard = self._get_or_create_shard(shard_key)

            # Check for existing duplicate in this shard via semantic similarity
            existing_node = None
            new_emb = embeddings[i] if i < len(embeddings) else []
            
            if new_emb:
                best_sim = 0
                for n in shard.nodes.values():
                    if n.is_super_node:
                        continue
                    sim = self._cosine_similarity(new_emb, n.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        existing_node = n
                
                if best_sim < 0.95:
                    existing_node = None

            if existing_node:
                # Merge into existing: update content if the new one is significantly different or longer?
                # For now, just boost metrics
                existing_node.salience = max(
                    existing_node.salience, mem.get("salience", 0.5)
                )
                existing_node.last_accessed = time.time()
                existing_node.access_count += 1
                print(f"   (Merged semantic duplicate into {existing_node.id}, sim: {best_sim:.3f})")
                continue

            node_id = self._generate_node_id()
            node = Node(
                id=node_id,
                content=content,
                embedding=embeddings[i] if i < len(embeddings) else [],
                type=mem.get("type", "semantic"),
                salience=mem.get("salience", 0.5),
                shard_key=shard_key,
            )

            shard.add_node(node)
            new_nodes.append((node_id, shard_key))

        self._link_within_shards(new_nodes)
        self._link_to_existing_memories(new_nodes)
        self._enforce_buffer_limit()

        if self.enable_hierarchy:
            for shard_key in set(sk for _, sk in new_nodes):
                if shard_key in self.shards:
                    shard = self.shards[shard_key]
                    if len(shard.nodes) > self.super_node_threshold:
                        self._create_super_nodes_for_shard(shard_key)

        elapsed = time.time() - start_time
        self.metrics["consolidation_times"].append(elapsed)
        print(f"✓ Background consolidation complete ({elapsed:.2f}s)")
        self._save_to_persistence()

    def _consolidate_to_buffer(self) -> str:
        consolidation_data = {
            "memories": self.short_term_memory.copy(),
            "timestamp": time.time(),
        }
        self.consolidation_queue.append(consolidation_data)
        self._async_consolidate()
        nodes, edges = self.buffer.size()
        return f"✓ Consolidation complete. Memory: {nodes} nodes, {edges} edges"

    def _link_within_shards(self, new_nodes: List[Tuple[str, str]]):
        shard_groups = defaultdict(list)
        for node_id, shard_key in new_nodes:
            shard_groups[shard_key].append(node_id)

        for shard_key, node_ids in shard_groups.items():
            if len(node_ids) < 2:
                continue

            shard = self.shards[shard_key]
            for i in range(len(node_ids) - 1):
                edge = Edge(
                    source=node_ids[i],
                    target=node_ids[i + 1],
                    weight=0.5,
                    edge_type="relates_to",
                )
                shard.add_edge(edge)

            for node_id in node_ids:
                node = shard.nodes[node_id]
                similarities = []
                for existing_id, existing_node in shard.nodes.items():
                    if existing_id == node_id or existing_id in node_ids:
                        continue
                    sim = self._cosine_similarity(
                        node.embedding, existing_node.embedding
                    )
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
                sim = self._cosine_similarity(
                    new_node.embedding, existing_node.embedding
                )
                similarities.append((existing_id, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            for existing_id, similarity in similarities[:3]:
                if similarity > 0.5:
                    existing_key_1 = (new_id, existing_id)
                    existing_key_2 = (existing_id, new_id)
                    edge_exists = False
                    for shard in self.shards.values():
                        if (
                            existing_key_1 in shard.edges
                            or existing_key_2 in shard.edges
                        ):
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
            print(f"✓ Created {links_created} cross-conversation links")

    def _create_super_nodes_for_shard(self, shard_key: str):
        shard = self.shards[shard_key]
        if len(shard.nodes) < self.super_node_threshold:
            return

        existing_super = [
            n for n in self.super_nodes.values() if n.shard_key == shard_key
        ]
        if existing_super:
            return

        print(
            f"  Creating super-node for shard '{shard_key}' ({len(shard.nodes)} nodes)"
        )
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
        print(f"  ✓ Created super-node {super_id} with {len(nodes)} children")

    def run_consolidation(
        self, weight_threshold: float = 0.6, merge_similar: bool = True
    ) -> str:
        """
        Performs a deep, graph-wide consolidation of memories.

        This is a more intensive process than the per-conversation consolidation.
        It identifies strongly connected components in the graph and extracts
        higher-level profile insights from them.

        Args:
            weight_threshold: Minimum association strength to consider for profile extraction.
            merge_similar: If True, execute a passes to deduplicate near-identical nodes (>0.95 sim).

        Returns:
            A summary of performed consolidation actions.

        Example:
            ```python
            # Force a deep consolidation
            results = ms.run_consolidation(merge_similar=True)
            print(results)
            ```
        """
        results = []
        print("🔄 Running consolidation...")

        if merge_similar:
            merged = self._merge_similar_nodes(similarity_threshold=0.95)
            if merged > 0:
                results.append(f"✓ Merged {merged} similar nodes")

        components = self.buffer.get_connected_components()
        profile_updates = 0

        for component in components:
            if len(component) < 3:
                continue

            component_edges = []
            for shard in self.shards.values():
                for (src, tgt), edge in shard.edges.items():
                    if src in component and tgt in component:
                        component_edges.append(edge)

            if not component_edges:
                continue

            avg_weight = sum(e.weight for e in component_edges) / len(component_edges)

            if avg_weight > 0.3:
                profile_update = self._extract_profile_from_component(component)
                if "Updated" in profile_update:
                    profile_updates += 1
                    results.append(profile_update)

        pruned = self.buffer.prune_weak_edges(threshold=self.prune_threshold)
        if pruned > 0:
            results.append(f"✓ Pruned {pruned} weak edges")

        if profile_updates > 0:
            results.append(f"✓ Updated {profile_updates} profile domains")
        else:
            all_contents = [
                node.content
                for node in self.buffer.nodes.values()
                if not node.is_super_node
            ]
            if len(all_contents) >= 3:
                profile_update = self._extract_profile_from_contents(all_contents)
                if "Updated" in profile_update:
                    results.append(profile_update)

        if not results:
            results.append("✓ No consolidation actions needed")
        return "\n".join(results)

    def _extract_profile_from_component(self, component: Set[str]) -> str:
        contents = []
        for nid in component:
            node = self.buffer.get_node(nid)
            if node and not node.is_super_node:
                contents.append(node.content)

        if not contents:
            return "No content to extract"
        return self._extract_profile_from_contents(contents)

    def _extract_profile_from_contents(self, contents: List[str]) -> str:
        if not contents:
            return "No content to extract"

        system_prompt = """Analyze these related memories and generate brief, factual personality insights (1-2 sentences each).
Identify all applicable domains: preferences, personality_traits, knowledge_domains, interaction_style, or key_experiences.
Return a JSON object where keys are the domain names and values are the specific insights.
Example: {"preferences": "User prefers Python for data science.", "knowledge_domains": "Exhibits deep expertise in memory systems."}"""

        prompt = "Related memories:\n" + "\n".join([f"- {c}" for c in contents[:10]])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self._call_llm(messages, response_format={"type": "json_object"})
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            data = json.loads(response)

            updated_any = False
            for domain, insight in data.items():
                if domain in self.profile.data and insight:
                    current = self.profile.data.get(domain, "")
                    if current and insight not in current:
                        updated = f"{current}. {insight}".strip()
                    else:
                        updated = insight

                    self.profile.update_domain(domain, updated)
                    print(f"  ✓ Profile updated: {domain} = {insight[:50]}...")
                    updated_any = True

            if updated_any:
                return f"✓ Updated profile domains"

        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON parse error: {e}")

        return "Failed to extract profile"

    def _merge_similar_nodes(self, similarity_threshold: float = 0.95) -> int:
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

            sim = self._cosine_similarity(node1.embedding, node2.embedding)
            if sim > similarity_threshold:
                node1.content = f"{node1.content} | {node2.content}"
                node1.salience = max(node1.salience, node2.salience)
                node1.access_count += node2.access_count

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
        return merged_count

    def get_stats(self) -> Dict:
        nodes, edges = self.buffer.size()
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
        avg_consolidation = (
            np.mean(self.metrics["consolidation_times"])
            if self.metrics["consolidation_times"]
            else 0
        )
        cache_hit_rate = self.query_cache.get_hit_rate() if self.query_cache else 0.0

        return {
            "buffer_nodes": nodes,
            "buffer_edges": edges,
            "num_shards": len(self.shards),
            "num_super_nodes": len(self.super_nodes),
            "short_term_memories": len(self.short_term_memory),
            "conversation_active": self.conversation_active,
            "conversation_count": self.conversation_count,
            "profile_domains_filled": sum(1 for v in self.profile.data.values() if v),
            "auto_consolidate": self.auto_consolidate,
            "performance": {
                "avg_retrieval_ms": f"{avg_retrieval:.1f}",
                "p95_retrieval_ms": f"{p95_retrieval:.1f}",
                "avg_consolidation_s": f"{avg_consolidation:.2f}",
                "cache_hit_rate": f"{cache_hit_rate:.1%}",
                "llm_calls": self.metrics["llm_calls"],
                "embedding_calls": self.metrics["embedding_calls"],
            },
        }

    def display_stats(self) -> str:
        stats = self.get_stats()
        next_consolidation = self.consolidate_every - (
            self.conversation_count % self.consolidate_every
        )
        return f"""
📊 SCALABLE MEMORY SYSTEM STATS:
STORAGE:
  • Buffer nodes: {stats["buffer_nodes"]} / {self.max_buffer_size} max
  • Buffer edges: {stats["buffer_edges"]}
  • Shards: {stats["num_shards"]}
  • Super-nodes: {stats["num_super_nodes"]}
  • STM: {stats["short_term_memories"]}
  • Conversations: {stats["conversation_count"]}
  • Profile domains: {stats["profile_domains_filled"]}/5

⚡ PERFORMANCE:
  • Avg retrieval: {stats["performance"]["avg_retrieval_ms"]}ms
  • P95 retrieval: {stats["performance"]["p95_retrieval_ms"]}ms
  • Avg consolidation: {stats["performance"]["avg_consolidation_s"]}s
  • Cache hit rate: {stats["performance"]["cache_hit_rate"]}
  • LLM calls: {stats["performance"]["llm_calls"]}
  • Embedding calls: {stats["performance"]["embedding_calls"]}

⚙️ AUTO-MANAGEMENT:
  • Auto-consolidate: {"ON" if stats["auto_consolidate"] else "OFF"} (every {self.consolidate_every})
    → Next in: {next_consolidation} conversation(s)
  • Auto-prune: {"ON" if self.auto_prune else "OFF"} (threshold: {self.prune_threshold})
  • Max buffer: {self.max_buffer_size} nodes
  • Sharding: {"ON" if self.enable_sharding else "OFF"}
  • Hierarchy: {"ON" if self.enable_hierarchy else "OFF"}
  • Caching: {"ON" if self.enable_caching else "OFF"}
  • Async: {"ON" if self.enable_async else "OFF"}
"""

    def display_memories(self, limit: int = 10) -> str:
        if not self.buffer.nodes:
            return "No memories stored yet."
        nodes = self.buffer.get_all_nodes_summary()
        output = [
            f"\n💭 Stored Memories (showing {min(limit, len(nodes))} of {len(nodes)}):"
        ]
        for i, node in enumerate(nodes[:limit], 1):
            output.append(
                f"\n{i}. [{node['type']}] 📦 {node['shard']} (salience: {node['salience']:.2f}, accessed: {node['access_count']}x)"
            )
            output.append(f"   {node['content']}")
        return "\n".join(output)

    def display_profile(self) -> str:
        return f"\n👤 User Profile:\n{self.profile.get_context()}\n"

    def save_state(self, filename: str = "memory_state.json") -> str:
        state = {
            "shards": {
                k: {
                    "nodes": [n.to_dict() for n in v.nodes.values()],
                    "edges": [e.to_dict() for e in v.edges.values()],
                }
                for k, v in self.shards.items()
            },
            "super_nodes": [n.to_dict() for n in self.super_nodes.values()],
            "profile": self.profile.to_dict(),
            "node_counter": self.node_counter,
            "conversation_count": self.conversation_count,
            "settings": {
                "auto_consolidate": self.auto_consolidate,
                "consolidate_every": self.consolidate_every,
                "auto_prune": self.auto_prune,
                "prune_threshold": self.prune_threshold,
                "max_buffer_size": self.max_buffer_size,
            },
        }
        with open(filename, "w") as f:
            json.dump(state, f, indent=2)
        return f"✓ State saved to {filename}"

    def load_state(self, filename: str = "memory_state.json") -> str:
        try:
            with open(filename, "r") as f:
                state = json.load(f)

            self.shards = {}
            for shard_key, shard_data in state.get("shards", {}).items():
                shard = MemoryShard(shard_key)
                for node_data in shard_data.get("nodes", []):
                    node = Node(**node_data)
                    shard.add_node(node)
                for edge_data in shard_data.get("edges", []):
                    edge = Edge(**edge_data)
                    shard.add_edge(edge)
                self.shards[shard_key] = shard

            self.super_nodes = {
                n["id"]: Node(**n) for n in state.get("super_nodes", [])
            }
            profile_data = state.get("profile", {})
            self.profile.data = profile_data.get("data", self.profile.data)
            self.profile.last_updated = profile_data.get("last_updated", time.time())

            self.node_counter = state.get("node_counter", 0)
            self.conversation_count = state.get("conversation_count", 0)

            settings = state.get("settings", {})
            for key, val in settings.items():
                if hasattr(self, key):
                    setattr(self, key, val)
            return f"✓ State loaded from {filename}"
        except FileNotFoundError:
            return f"⚠ File {filename} not found"

    def _save_to_persistence(self):
        state = {
            "shards": self.shards,
            "super_nodes": self.super_nodes,
            "profile": self.profile,
            "node_counter": self.node_counter,
            "conversation_count": self.conversation_count,
            "query_cache": self.query_cache.cache if self.query_cache else None,
            "settings": {
                "enable_sharding": self.enable_sharding,
                "enable_hierarchy": self.enable_hierarchy,
                "enable_caching": self.enable_caching,
                "enable_async": self.enable_async,
                "max_shard_size": self.max_shard_size,
                "super_node_threshold": self.super_node_threshold,
                "auto_consolidate": self.auto_consolidate,
                "consolidate_every": self.consolidate_every,
                "auto_prune": self.auto_prune,
                "prune_threshold": self.prune_threshold,
                "max_buffer_size": self.max_buffer_size,
            },
        }
        if self.persistence.save(state):
            pass  # Silent save or debug log

    def _load_from_persistence(self):
        state = self.persistence.load()
        if not state:
            return

        self.shards = state.get("shards", {})
        self.super_nodes = state.get("super_nodes", {})
        self.profile = state.get("profile", Profile())
        self.node_counter = state.get("node_counter", 0)
        self.conversation_count = state.get("conversation_count", 0)

        # Re-initialize buffer with loaded shards
        self.buffer = BufferGraph(self.shards, self.super_nodes)

        if self.query_cache and state.get("query_cache"):
            self.query_cache.cache = state.get("query_cache")

        print(f"✓ Restored state from disk ({len(self.buffer.nodes)} nodes)")
