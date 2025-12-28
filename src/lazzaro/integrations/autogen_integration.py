
from typing import Dict, List, Optional, Union, Any
from ..core.memory_system import MemorySystem

class LazzaroAutogenAgent:
    """
    An extension/mixin for Autogen agents to use Lazzaro memory.
    """
    def __init__(self, agent: Any, memory_system: MemorySystem):
        self.agent = agent
        self.memory_system = memory_system
        self._setup_hooks()

    def _setup_hooks(self):
        # Hooks into the agent's message processing
        # This is a bit tricky with Autogen as it depends on the version
        # but usually we can register a reply function.
        
        try:
            from autogen import Agent, ConversableAgent
            if isinstance(self.agent, ConversableAgent):
                self.agent.register_reply(
                    [Agent, None],
                    reply_func=self._generate_memory_aware_reply,
                    position=0 # Run before other reply functions
                )
        except ImportError:
            print("⚠ Autogen not installed. Integration may not work.")

    def _generate_memory_aware_reply(
        self,
        recipient: Any,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> Union[str, Dict, None]:
        if not messages:
            return None
        
        last_message = messages[-1].get("content", "")
        if not last_message:
            return None

        # 1. Retrieve context from Lazzaro
        query_emb = self.memory_system._get_embedding(last_message)
        retrieved_ids = self.memory_system._optimized_retrieval(query_emb, last_message)
        
        context_parts = []
        profile_context = self.memory_system.profile.get_context()
        if profile_context and profile_context != "No profile data yet.":
            context_parts.append(f"User Profile: {profile_context}")

        if retrieved_ids:
            memory_texts = []
            for nid in retrieved_ids:
                node = self.memory_system.buffer.get_node(nid)
                if node:
                    memory_texts.append(node.content)
            if memory_texts:
                context_parts.append("Relevant Context:\n" + "\n".join(memory_texts))

        if context_parts:
            # Inject context into the system message or the last message
            context_str = "\n\n[LAZZARO MEMORY CONTEXT]\n" + "\n\n".join(context_parts)
            
            # We modify the messages in-place or return a modified prompt
            # For Autogen, we can't easily modify the message history here without side effects
            # Better to return None (meaning continue to next reply function) but with context added to the agent's system message
            
            original_system_msg = self.agent.system_message
            if "[LAZZARO MEMORY CONTEXT]" not in original_system_msg:
                self.agent.update_system_message(original_system_msg + context_str)
            else:
                # Update existing context
                import re
                new_system_msg = re.sub(r"\[LAZZARO MEMORY CONTEXT\].*$", context_str, original_system_msg, flags=re.DOTALL)
                self.agent.update_system_message(new_system_msg)

        # 2. Record the exchange in Lazzaro after a reply is generated
        # We'll use a post-reply hook if possible, but Autogen handles saving history themselves.
        # We might need to hook into the agent's `receive` method to catch the user input.
        
        if not self.memory_system.conversation_active:
            self.memory_system.start_conversation()
        
        self.memory_system.add_to_short_term(last_message, "episodic", salience=0.7)
        self.memory_system.conversation_history.append({"role": "user", "content": last_message})

        return None # Let Autogen's default reply generation handle it
