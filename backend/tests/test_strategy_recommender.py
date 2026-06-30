"""Tests for strategy recommender."""
import pytest
from app.services.strategy_recommender import strategy_recommender


class TestStrategyRecommendation:
    def test_vector_threshold(self):
        """Complexity < 0.3 should recommend vector."""
        result = strategy_recommender.recommend(0.29)
        assert result["recommended_strategy"] == "vector"
        assert result["recommended_strategy_name"] == "向量检索"

    def test_hybrid_threshold(self):
        """Complexity in [0.3, 0.7) should recommend hybrid."""
        result = strategy_recommender.recommend(0.30)
        assert result["recommended_strategy"] == "hybrid"
        assert result["recommended_strategy_name"] == "混合检索"

        result = strategy_recommender.recommend(0.69)
        assert result["recommended_strategy"] == "hybrid"

    def test_graph_threshold(self):
        """Complexity >= 0.7 should recommend graph."""
        result = strategy_recommender.recommend(0.70)
        assert result["recommended_strategy"] == "graph"
        assert result["recommended_strategy_name"] == "图检索"

    def test_reason_generation(self):
        result = strategy_recommender.recommend(
            0.5,
            query="test",
            features={"question_type": "comparative", "entity_count": 3, "relation_count": 2},
        )
        assert len(result["reason"]) > 0
        assert "comparative" in result["reason"] or "混合" in result["reason"]

    def test_alternatives(self):
        result = strategy_recommender.recommend(0.5)
        assert len(result["alternatives"]) == 2
        alt_ids = [a["id"] for a in result["alternatives"]]
        assert "vector" in alt_ids
        assert "graph" in alt_ids

    def test_complexity_included(self):
        result = strategy_recommender.recommend(0.5)
        assert result["complexity_score"] == 0.5
