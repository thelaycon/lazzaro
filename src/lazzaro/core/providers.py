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

class GeminiLLM(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model_name = model
            self.model = genai.GenerativeModel(model)
        except ImportError:
            print("⚠ Error: 'google-generativeai' not installed. Run 'pip install google-generativeai'")
            self.model = None

    def completion(self, messages: List[Dict[str, str]], response_format: Dict = None) -> str:
        if not self.model: return ""
        try:
            # Convert messages to Gemini format
            prompt = ""
            for msg in messages:
                role = "User" if msg['role'] == 'user' else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"⚠ Gemini error: {e}")
            return ""

    def completion_stream(self, messages: List[Dict[str, str]], response_format: Dict = None):
        if not self.model: yield ""; return
        try:
            prompt = ""
            for msg in messages:
                role = "User" if msg['role'] == 'user' else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
            
            response = self.model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"⚠ Gemini stream error: {e}")
            yield ""

class GeminiEmbedder(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "models/embedding-001"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = model
            self.genai = genai
        except ImportError:
            print("⚠ Error: 'google-generativeai' not installed.")
            self.model = None

    def embed(self, text: str) -> List[float]:
        if not self.model: return [0.0] * 768
        try:
            result = self.genai.embed_content(model=self.model, content=text)
            return result['embedding']
        except Exception as e:
            print(f"⚠ Gemini Embedding error: {e}")
            return [0.0] * 768

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        if not self.model: return [[0.0] * 768 for _ in texts]
        try:
            result = self.genai.embed_content(model=self.model, content=texts)
            return result['embedding']
        except Exception as e:
            print(f"⚠ Gemini Batch Embedding error: {e}")
            return [[0.0] * 768 for _ in texts]

class TogetherLLM(LLMProvider):
    def __init__(self, api_key: str, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        try:
            from together import Together
            self.client = Together(api_key=api_key)
            self.model = model
        except ImportError:
            print("⚠ Error: 'together' not installed. Run 'pip install together'")
            self.client = None

    def completion(self, messages: List[Dict[str, str]], response_format: Dict = None) -> str:
        if not self.client: return ""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠ Together error: {e}")
            return ""

    def completion_stream(self, messages: List[Dict[str, str]], response_format: Dict = None):
        if not self.client: yield ""; return
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            print(f"⚠ Together stream error: {e}")
            yield ""

class TogetherEmbedder(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "togethercomputer/m2-bert-80m-8k-retrieval"):
        try:
            from together import Together
            self.client = Together(api_key=api_key)
            self.model = model
        except ImportError:
            print("⚠ Error: 'together' not installed.")
            self.client = None

    def embed(self, text: str) -> List[float]:
        if not self.client: return [0.0] * 768
        try:
            response = self.client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        except Exception as e:
            print(f"⚠ Together Embedding error: {e}")
            return [0.0] * 768

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        if not self.client: return [[0.0] * 768 for _ in texts]
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"⚠ Together Batch Embedding error: {e}")
            return [[0.0] * 768 for _ in texts]
