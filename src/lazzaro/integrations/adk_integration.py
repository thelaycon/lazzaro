
from typing import Any, Dict, List, Optional
from ..core.memory_system import MemorySystem

class LazzaroADKPlugin:
    """
    Integration for Google Agent Development Kit (ADK).
    """
    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system

    def as_tool(self):
        """Returns Lazzaro retrieval as an ADK-compatible tool."""
        return {
            "name": "lazzaro_memory_retrieval",
            "description": "Retrieve relevant past memories and user profile information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The current user query to find relevant memories for."
                    }
                },
                "required": ["query"]
            },
            "func": self.retrieve
        }

    def retrieve(self, query: str) -> str:
        # Internal retrieval logic
        query_emb = self.memory_system._get_embedding(query)
        retrieved_ids = self.memory_system._optimized_retrieval(query_emb, query)
        
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
                context_parts.append("Relevant Memories:\n" + "\n".join(memory_texts))

        return "\n\n".join(context_parts) if context_parts else "No relevant memories found."

    def observe(self, user_input: str, agent_output: str):
        """Records a conversation turn in Lazzaro."""
        if not self.memory_system.conversation_active:
            self.memory_system.start_conversation()
        
        self.memory_system.add_to_short_term(user_input, "episodic", salience=0.7)
        self.memory_system.conversation_history.append({"role": "user", "content": user_input})
        
        self.memory_system.add_to_short_term(agent_output, "semantic", salience=0.5)
        self.memory_system.conversation_history.append({"role": "assistant", "content": agent_output})
