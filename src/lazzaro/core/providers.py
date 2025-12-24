import openai
from typing import List, Dict, Any
from .interfaces import LLMProvider, EmbeddingProvider

class OpenAILLM(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def completion(self, messages: List[Dict[str, str]], response_format: Dict = None) -> str:
        try:
            kwargs = {"model": self.model, "messages": messages, "temperature": 0.7}
            if response_format:
                kwargs["response_format"] = response_format
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"⚠ LLM error: {e}")
            return ""

    def completion_stream(self, messages: List[Dict[str, str]], response_format: Dict = None):
        try:
            kwargs = {"model": self.model, "messages": messages, "temperature": 0.7, "stream": True}
            if response_format:
                kwargs["response_format"] = response_format
            
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            print(f"⚠ LLM stream error: {e}")
            yield ""

class OpenAIEmbedder(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def embed(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        except Exception as e:
            print(f"⚠ Embedding error: {e}")
            return [0.0] * 1536

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"⚠ Batch embedding error: {e}")
            return [[0.0] * 1536 for _ in texts]
