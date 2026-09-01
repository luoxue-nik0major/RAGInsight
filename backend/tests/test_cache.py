"""Tests for cache layer."""
from app.services.cache import QueryCache, AnswerCache


class TestQueryCacheCollectionKey:
    def test_same_query_different_collection_not_hit(self):
        """Same query+strategy in a different collection must not hit the cache."""
        cache = QueryCache()
        cache.set("李白是谁", "vector", {"chunks": [{"content": "x"}]}, collection="chinese_poetry")
        assert cache.get("李白是谁", "vector", "chinese_poetry") is not None
        assert cache.get("李白是谁", "vector", "nq_documents") is None

    def test_collection_none_and_empty_equivalent(self):
        """None and '' normalize to the same key segment."""
        cache = QueryCache()
        cache.set("q", "vector", {"chunks": []})
        assert cache.get("q", "vector", None) is not None
        assert cache.get("q", "vector") is not None

    def test_different_strategy_not_hit(self):
        cache = QueryCache()
        cache.set("q", "vector", {"chunks": []}, collection="c1")
        assert cache.get("q", "hybrid", "c1") is None


class TestAnswerCacheCollectionKey:
    def test_same_query_different_collection_not_hit(self):
        cache = AnswerCache()
        cache.set("法国的首都是哪里", "vector", "巴黎", collection="chinese_poetry")
        assert cache.get("法国的首都是哪里", "vector", "chinese_poetry") == "巴黎"
        assert cache.get("法国的首都是哪里", "vector", "nq_documents") is None
