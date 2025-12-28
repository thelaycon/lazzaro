import openai
import logging
from typing import List, Dict, Any
from .interfaces import LLMProvider, EmbeddingProvider
from .resilience import CircuitBreaker, RetryManager, FallbackManager


class ResilientOpenAILLM(LLMProvider):
    """OpenAI LLM provider with circuit breaker and retry logic."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", config=None):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.config = config
        
        # Setup resilience components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=getattr(config, 'circuit_breaker_threshold', 5),
            timeout=getattr(config, 'circuit_breaker_timeout', 60)
        )
        self.retry_manager = RetryManager(
            max_retries=getattr(config, 'max_retries', 3),
            backoff_factor=getattr(config, 'retry_backoff_factor', 2.0)
        )
        self.fallback_manager = FallbackManager()
        self.logger = logging.getLogger(__name__)

    def completion(self, messages: List[Dict[str, str]], response_format: Dict = None) -> str:
        """Generate completion with resilience."""
        def _do_completion():
            kwargs = {"model": self.model, "messages": messages, "temperature": 0.7}
            if response_format:
                kwargs["response_format"] = response_format
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        
        try:
            return self.retry_manager.with_retry(
                lambda: self.circuit_breaker.call(_do_completion)
            )
        except Exception as e:
            self.logger.error(f"OpenAI LLM completion failed: {e}")
            return self.fallback_manager.get_fallback_response("general")

    def completion_stream(self, messages: List[Dict[str, str]], response_format: Dict = None):
        """Generate streaming completion with resilience."""
        def _do_stream():
            kwargs = {"model": self.model, "messages": messages, "temperature": 0.7, "stream": True}
            if response_format:
                kwargs["response_format"] = response_format
            
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        
        try:
            # For streaming, we'll use retry but skip circuit breaker for simplicity
            return self.retry_manager.with_retry(_do_stream)
        except Exception as e:
            self.logger.error(f"OpenAI LLM stream failed: {e}")
            yield self.fallback_manager.get_fallback_response("general")


class ResilientOpenAIEmbedder(EmbeddingProvider):
    """OpenAI embedding provider with resilience."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", config=None):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.config = config
        
        # Setup resilience components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=getattr(config, 'circuit_breaker_threshold', 5),
            timeout=getattr(config, 'circuit_breaker_timeout', 60)
        )
        self.retry_manager = RetryManager(
            max_retries=getattr(config, 'max_retries', 3),
            backoff_factor=getattr(config, 'retry_backoff_factor', 2.0)
        )
        self.fallback_manager = FallbackManager()
        self.logger = logging.getLogger(__name__)

    def embed(self, text: str) -> List[float]:
        """Generate embedding with resilience."""
        def _do_embed():
            response = self.client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        
        try:
            return self.retry_manager.with_retry(
                lambda: self.circuit_breaker.call(_do_embed)
            )
        except Exception as e:
            self.logger.error(f"OpenAI embedding failed: {e}")
            return self.fallback_manager.get_fallback_embedding(1536)

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings with resilience."""
        if not texts:
            return []
            
        def _do_batch_embed():
            response = self.client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        
        try:
            return self.retry_manager.with_retry(
                lambda: self.circuit_breaker.call(_do_batch_embed)
            )
        except Exception as e:
            self.logger.error(f"OpenAI batch embedding failed: {e}")
            return [self.fallback_manager.get_fallback_embedding(1536) for _ in texts]


# Factory function to create resilient providers
def create_resilient_providers(config, openai_api_key: str = None):
    """Factory to create resilient provider instances."""
    if not openai_api_key:
        raise ValueError("OpenAI API key required for resilient providers")
    
    llm = ResilientOpenAILLM(
        api_key=openai_api_key,
        model=config.llm_model,
        config=config
    )
    
    embedder = ResilientOpenAIEmbedder(
        api_key=openai_api_key,
        model=config.embedding_model,
        config=config
    )
    
    return llm, embedder