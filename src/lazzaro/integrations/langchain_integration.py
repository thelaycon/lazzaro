
from typing import Any, Dict, List, Optional
from langchain_core.memory import BaseMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from ..core.memory_system import MemorySystem

class LazzaroLangChainMemory(BaseMemory):
    """Lazzaro memory integration for LangChain."""
    
    memory_system: MemorySystem
    memory_key: str = "history"
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    return_messages: bool = False

    def __init__(self, memory_system: MemorySystem, **kwargs):
        super().__init__(memory_system=memory_system, **kwargs)

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Retrieve relevant memories from Lazzaro based on the current input
        user_message = inputs.get(self.input_key) or inputs.get("input") or ""
        if not user_message:
            return {self.memory_key: "" if not self.return_messages else []}

        # Use Lazzaro's internal retrieval logic to get context
        # We don't call chat() because that calls the LLM. 
        # We just want the context part.
        query_emb = self.memory_system._get_embedding(user_message)
        retrieved_ids = self.memory_system._optimized_retrieval(query_emb, user_message)
        
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
                context_parts.append("Relevant Past Memories:\n" + "\n".join(memory_texts))

        context = "\n\n".join(context_parts)

        if self.return_messages:
            return {self.memory_key: [AIMessage(content=context)] if context else []}
        return {self.memory_key: context}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        user_input = inputs.get(self.input_key) or inputs.get("input") or ""
        ai_output = outputs.get(self.output_key) or outputs.get("output") or ""
        
        if not self.memory_system.conversation_active:
            self.memory_system.start_conversation()
            
        if user_input:
            self.memory_system.add_to_short_term(user_input, "episodic", salience=0.7)
            self.memory_system.conversation_history.append({"role": "user", "content": user_input})
            
        if ai_output:
            self.memory_system.add_to_short_term(ai_output, "semantic", salience=0.5)
            self.memory_system.conversation_history.append({"role": "assistant", "content": ai_output})

    def clear(self) -> None:
        self.memory_system.end_conversation()
