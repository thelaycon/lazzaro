
from typing import Dict, List, Optional, Any, Annotated
from ..core.memory_system import MemorySystem

class LazzaroLangGraph:
    """
    Utilities for using Lazzaro within LangGraph workflows.
    """
    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system

    def get_memory_node(self):
        """
        Returns a LangGraph node that retrieves relevant memories and adds them to the state.
        """
        def memory_node(state: Dict[str, Any]):
            # Assume state has a "messages" key or "input" key
            messages = state.get("messages", [])
            if not messages:
                user_msg = state.get("input", "")
            else:
                user_msg = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

            if not user_msg:
                return {"lazzaro_context": ""}

            query_emb = self.memory_system._get_embedding(user_msg)
            retrieved_ids = self.memory_system._optimized_retrieval(query_emb, user_msg)
            
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
                    context_parts.append("Past Memories:\n" + "\n".join(memory_texts))

            return {"lazzaro_context": "\n\n".join(context_parts)}

        return memory_node

    def get_record_node(self):
        """
        Returns a LangGraph node that records the latest interaction in Lazzaro.
        """
        def record_node(state: Dict[str, Any]):
            messages = state.get("messages", [])
            if len(messages) < 2:
                return {}

            user_msg = messages[-2].content if hasattr(messages[-2], "content") else str(messages[-2])
            ai_msg = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

            if not self.memory_system.conversation_active:
                self.memory_system.start_conversation()

            self.memory_system.add_to_short_term(user_msg, "episodic", salience=0.7)
            self.memory_system.conversation_history.append({"role": "user", "content": user_msg})
            
            self.memory_system.add_to_short_term(ai_msg, "semantic", salience=0.5)
            self.memory_system.conversation_history.append({"role": "assistant", "content": ai_msg})

            return {}

        return record_node
