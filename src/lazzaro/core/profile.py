import time

class Profile:
    def __init__(self):
        self.data = {
            "preferences": "",
            "personality_traits": "",
            "knowledge_domains": "",
            "interaction_style": "",
            "key_experiences": ""
        }
        self.last_updated = time.time()

    def update_domain(self, domain: str, content: str):
        if domain in self.data:
            self.data[domain] = content
            self.last_updated = time.time()

    def get_context(self) -> str:
        parts = []
        for domain, content in self.data.items():
            if content:
                parts.append(f"{domain.replace('_', ' ').title()}: {content}")
        return "\n".join(parts) if parts else "No profile data yet."

    def to_dict(self):
        return {
            "data": self.data,
            "last_updated": self.last_updated
        }
