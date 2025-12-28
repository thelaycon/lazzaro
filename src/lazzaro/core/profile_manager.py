import time
import logging
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

from ..models.graph import Node, Edge
from .interfaces import LLMProvider
from .profile import Profile


class ProfileManager:
    """Focused on user profile evolution and insights extraction."""
    
    def __init__(self, config: 'MemoryConfig', llm: LLMProvider, profile: Profile):
        self.config = config
        self.llm = llm
        self.profile = profile
        self.logger = logging.getLogger(__name__)
        self.metrics = {"profile_updates": 0, "insight_calls": 0}
    
    def extract_profile_from_component(self, component: Set[str], buffer) -> str:
        """Extract profile insights from a connected component."""
        contents = []
        for nid in component:
            node = buffer.get_node(nid)
            if node and not node.is_super_node:
                contents.append(node.content)
        
        if not contents:
            return "No content to extract"
        
        return self.extract_profile_from_contents(contents)
    
    def extract_profile_from_contents(self, contents: List[str]) -> str:
        """Extract profile insights from a list of memory contents."""
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
        
        try:
            self.metrics["insight_calls"] += 1
            response = self.llm.completion(messages, response_format={"type": "json_object"})
            
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            
            import json
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
                    self.logger.info(f"Profile updated: {domain} = {insight[:50]}...")
                    updated_any = True
            
            if updated_any:
                self.metrics["profile_updates"] += 1
                return "✓ Updated profile domains"
            
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Profile extraction failed: {e}")
        
        return "Failed to extract profile"
    
    def get_comprehensive_insights(self, buffer) -> str:
        """Generate comprehensive user insights using LLM analysis."""
        # Collect all non-super-node memories
        all_contents = []
        for node in buffer.nodes.values():
            if not node.is_super_node:
                all_contents.append(node.content)
        
        if len(all_contents) < 3:
            return "Insufficient memories for comprehensive analysis"
        
        observations_json = json.dumps([{"content": c} for c in all_contents[:50]])
        
        system_prompt = f"""Analyze these atomic memories for user and provide a comprehensive psychological and knowledge profile. 
Identify long-term patterns, core beliefs, persistent interests, and significant life events reflected in the data.

Structure your response as:
1. **Personality Traits**: Key characteristics detected.
2. **Core Interests & Knowledge**: What the user knows and cares about.
3. **Behavioral Patterns**: How the user typically interacts or works.
4. **Recent Focus**: Most salient topics from recent memories.

Be clinical yet insightful. Do not include conversational filler."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Observations:\n{observations_json}"}
        ]
        
        try:
            self.metrics["insight_calls"] += 1
            return self.llm.completion(messages)
        except Exception as e:
            self.logger.error(f"Comprehensive insights failed: {e}")
            return "Unable to generate insights at this time"
    
    def get_context(self) -> str:
        """Get formatted profile context for LLM prompts."""
        return self.profile.get_context()
    
    def update_profile_domain(self, domain: str, insight: str):
        """Update a specific profile domain."""
        self.profile.update_domain(domain, insight)
        self.metrics["profile_updates"] += 1
    
    def get_stats(self) -> Dict:
        """Get profile management statistics."""
        return {
            "profile_updates": self.metrics["profile_updates"],
            "insight_calls": self.metrics["insight_calls"],
            "domains_filled": sum(1 for v in self.profile.data.values() if v),
            "total_domains": len(self.profile.data)
        }