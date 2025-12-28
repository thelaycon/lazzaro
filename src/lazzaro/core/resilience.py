import time
import logging
from enum import Enum
from typing import Callable, Any, Optional


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN" 
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Circuit breaker pattern for resilient API calls."""
    
    def __init__(
        self, 
        failure_threshold: int = 5, 
        timeout: int = 60,
        logger: Optional[logging.Logger] = None
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED
        self.logger = logger or logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN - calls blocked")
        
        try:
            result = func(*args, **kwargs)
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.logger.info("Circuit breaker transitioning to CLOSED")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
            
            raise e


class RetryManager:
    """Retry logic with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        logger: Optional[logging.Logger] = None
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logger or logging.getLogger(__name__)
    
    def with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Function failed after {self.max_retries} attempts: {e}")
                    raise e
                
                wait_time = self.backoff_factor ** attempt
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
        
        if last_exception:
            raise last_exception
        else:
            raise Exception("Unknown error in retry logic")


class FallbackManager:
    """Fallback strategies for provider failures."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def get_fallback_response(self, context: str = "general") -> str:
        """Get fallback response based on context."""
        fallbacks = {
            "general": "I'm experiencing technical difficulties. Please try again later.",
            "memory_consolidation": "",
            "embedding": [0.0] * 1536,  # Default OpenAI embedding size
            "profile_extraction": "{}"
        }
        return fallbacks.get(context, fallbacks["general"])
    
    def get_fallback_embedding(self, size: int = 1536) -> list:
        """Get fallback embedding vector."""
        return [0.0] * size