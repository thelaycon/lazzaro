from collections import OrderedDict
import threading
import hashlib
import time
from typing import List, Optional

class QueryCache:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    def get_embedding(self, query: str) -> Optional[List[float]]:
        with self.lock:
            key = self._hash_query(query)
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]['embedding']
            self.misses += 1
            return None

    def set_embedding(self, query: str, embedding: List[float]):
        with self.lock:
            key = self._hash_query(query)
            if key in self.cache:
                self.cache[key]['embedding'] = embedding
                self.cache.move_to_end(key)
            else:
                self.cache[key] = {'embedding': embedding, 'timestamp': time.time()}
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def get_results(self, query: str) -> Optional[List[str]]:
        with self.lock:
            key = self._hash_query(query)
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key].get('results')
            self.misses += 1
            return None

    def set_results(self, query: str, results: List[str]):
        with self.lock:
            key = self._hash_query(query)
            if key in self.cache:
                self.cache[key]['results'] = results
            else:
                self.cache[key] = {'results': results, 'timestamp': time.time()}

    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
