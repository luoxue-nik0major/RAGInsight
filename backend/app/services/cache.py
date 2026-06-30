"""Cache layer for RAG pipeline.
Uses in-memory TTLCache for fast retrieval without external dependencies.
"""
from typing import Dict, Any, Optional
from cachetools import TTLCache


class QueryCache:
    """Cache retrieval results by query text."""

    def __init__(self, maxsize: int = 100, ttl: int = 300):
        self._cache: Dict[str, Any] = TTLCache(maxsize=maxsize, ttl=ttl)

    def get(self, query: str, strategy: str = "vector") -> Optional[Dict[str, Any]]:
        key = f"{strategy}:{query}"
        return self._cache.get(key)

    def set(self, query: str, strategy: str, result: Dict[str, Any]):
        key = f"{strategy}:{query}"
        self._cache[key] = result

    def clear(self):
        self._cache.clear()


class AnswerCache:
    """Cache final answers for simple factual queries."""

    def __init__(self, maxsize: int = 100, ttl: int = 600):
        self._cache: Dict[str, Any] = TTLCache(maxsize=maxsize, ttl=ttl)

    def get(self, query: str, strategy: str = "vector") -> Optional[str]:
        key = f"{strategy}:{query}"
        return self._cache.get(key)

    def set(self, query: str, strategy: str, answer: str):
        key = f"{strategy}:{query}"
        self._cache[key] = answer

    def clear(self):
        self._cache.clear()


class EmbeddingCache:
    """Cache embedding vectors to avoid repeated ONNX inference."""

    def __init__(self, maxsize: int = 200, ttl: int = 300):
        self._cache: Dict[str, Any] = TTLCache(maxsize=maxsize, ttl=ttl)

    def get(self, text: str) -> Optional[Any]:
        return self._cache.get(text)

    def set(self, text: str, embedding: Any):
        self._cache[text] = embedding

    def clear(self):
        self._cache.clear()


# Global cache instances
query_cache = QueryCache()
answer_cache = AnswerCache()
embedding_cache = EmbeddingCache()
