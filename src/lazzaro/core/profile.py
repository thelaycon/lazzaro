import time

class Profile:
    """
    Manages the evolved multi-domain persona of the user.

    The Profile aggregates facts into high-level traits across five domains,
    providing a structured context for the LLM during interactions.

    Attributes:
        data: Dictionary containing the five profile domains.
        last_updated: Unix timestamp of the last domain update.

    Example:
        ```python
        profile = Profile()
        profile.update_domain("preferences", "User loves minimalist design.")
        print(profile.get_context())
        ```
    """

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
        """Updates a specific domain with new synthesized content."""
        if domain in self.data:
            self.data[domain] = content
            self.last_updated = time.time()

    def get_context(self) -> str:
        """Returns a formatted string of and all non-empty profile domains."""
        parts = []
        for domain, content in self.data.items():
            if content:
                parts.append(f"{domain.replace('_', ' ').title()}: {content}")
        return "\n".join(parts) if parts else "No profile data yet."

    def to_dict(self):
        """Returns a serializable dictionary of the profile."""
        return {
            "data": self.data,
            "last_updated": self.last_updated
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Creates a Profile from a dictionary."""
        instance = cls()
        instance.data = data.get("data", instance.data)
        instance.last_updated = data.get("last_updated", instance.last_updated)
        return instance
