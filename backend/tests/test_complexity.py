"""Tests for complexity analyzer."""
import pytest
from app.services.complexity import complexity_analyzer


class TestEnglishQueries:
    def test_factual_simple(self):
        """Simple factual query should have low complexity."""
        result = complexity_analyzer.analyze("Where is France?")
        assert result["question_type"] == "factual"
        assert result["complexity_score"] < 0.3

    def test_definition(self):
        result = complexity_analyzer.analyze("What is machine learning?")
        assert result["question_type"] == "definition"
        assert result["complexity_score"] >= 0.05

    def test_comparative(self):
        """Comparative query should be recognized and have moderate complexity."""
        result = complexity_analyzer.analyze("Compare machine learning and deep learning.")
        assert result["question_type"] == "comparative"
        assert result["complexity_score"] >= 0.2

    def test_causal(self):
        result = complexity_analyzer.analyze("Why does climate change happen?")
        assert result["question_type"] == "causal"
        assert result["complexity_score"] >= 0.1

    def test_multi_hop(self):
        result = complexity_analyzer.analyze("How does A relate to B? What is the impact on C? And why does it matter?")
        assert result["question_type"] == "multi_hop"
        assert result["complexity_score"] >= 0.1

    def test_empty_query(self):
        result = complexity_analyzer.analyze("")
        assert result["complexity_score"] == 0.0
        assert result["question_type"] == "factual"


class TestChineseQueries:
    def test_chinese_factual(self):
        result = complexity_analyzer.analyze("法国的首都是什么？")
        assert result["question_type"] in ("factual", "definition")
        assert result["complexity_score"] < 0.5

    def test_chinese_definition(self):
        result = complexity_analyzer.analyze("什么是机器学习？")
        assert result["question_type"] == "definition"

    def test_chinese_comparative(self):
        result = complexity_analyzer.analyze("比较李白和杜甫的诗歌风格")
        assert result["question_type"] == "comparative"
        assert result["complexity_score"] >= 0.2

    def test_chinese_causal(self):
        result = complexity_analyzer.analyze("为什么气候变化会发生？")
        assert result["question_type"] == "causal"

    def test_chinese_multi_hop(self):
        result = complexity_analyzer.analyze("李白和杜甫的诗歌风格有什么不同？他们对后世有什么影响？")
        assert result["question_type"] == "multi_hop"

    def test_chinese_features(self):
        result = complexity_analyzer.analyze("李白和杜甫的诗歌风格有什么不同？")
        features = result["features"]
        assert features["length"] > 0
        assert features["sentence_count"] >= 1
        assert "entity_count" in features
        assert features["entity_count"] >= 1  # 李白 / 杜甫 should be detected

    def test_chinese_conjunctions_counted(self):
        """Chinese conjunctions must be counted (no word-boundary regex for CJK)."""
        result = complexity_analyzer.analyze("李白和杜甫与白居易的诗歌")
        assert result["features"]["hop_demand_score"] > 0
