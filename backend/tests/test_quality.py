"""Tests for quality evaluator."""
import pytest
from app.services.quality import quality_evaluator


class TestEvaluateRelevance:
    def test_empty_chunks(self):
        """Relevance of empty chunk list should be 0."""
        result = quality_evaluator.evaluate_relevance([])
        assert result == 0.0

    def test_single_chunk(self):
        """Single chunk relevance should equal its score."""
        chunks = [{"relevance_score": 0.8}]
        result = quality_evaluator.evaluate_relevance(chunks)
        assert result == pytest.approx(0.8, 0.01)

    def test_multiple_chunks(self):
        """Average relevance across chunks."""
        chunks = [
            {"relevance_score": 0.9},
            {"relevance_score": 0.7},
            {"relevance_score": 0.8},
        ]
        result = quality_evaluator.evaluate_relevance(chunks)
        assert result == pytest.approx(0.8, 0.01)


class TestEvaluateDiversity:
    def test_empty_chunks(self):
        """Empty chunks trivially diverse (no pairs to compare)."""
        result = quality_evaluator.evaluate_diversity([])
        assert result == 1.0

    def test_single_chunk(self):
        """Single chunk is trivially diverse."""
        chunks = [{"content": "hello world"}]
        result = quality_evaluator.evaluate_diversity(chunks)
        assert result == 1.0

    def test_identical_chunks(self):
        """Identical chunks should have 0 diversity."""
        chunks = [
            {"content": "hello world"},
            {"content": "hello world"},
        ]
        result = quality_evaluator.evaluate_diversity(chunks)
        assert result == 0.0

    def test_different_chunks(self):
        """Different chunks should have positive diversity."""
        chunks = [
            {"content": "machine learning artificial intelligence"},
            {"content": "natural language processing deep learning"},
        ]
        result = quality_evaluator.evaluate_diversity(chunks)
        assert result > 0.0
        assert result <= 1.0


class TestEvaluateCoverage:
    def test_empty_query(self):
        result = quality_evaluator.evaluate_coverage("", [])
        assert result == 0.0

    def test_full_coverage(self):
        """All query entities appear in chunks."""
        query = "Paris France"
        chunks = [
            {"content": "Paris is the capital of France"},
        ]
        result = quality_evaluator.evaluate_coverage(query, chunks)
        assert result == 1.0

    def test_partial_coverage(self):
        """Some entities missing from chunks."""
        query = "Paris France Germany"
        chunks = [
            {"content": "Paris is the capital of France"},
        ]
        result = quality_evaluator.evaluate_coverage(query, chunks)
        assert 0.0 < result < 1.0

    def test_no_coverage(self):
        """No query entities in chunks."""
        query = "Tokyo Japan"
        chunks = [
            {"content": "Paris is the capital of France"},
        ]
        result = quality_evaluator.evaluate_coverage(query, chunks)
        assert result == 0.0


class TestEvaluateAll:
    def test_empty_chunks(self):
        result = quality_evaluator.evaluate_all("test", [])
        assert result["relevance"] == 0.0
        assert result["diversity"] == 1.0  # trivially diverse
        assert result["coverage"] == 0.0  # empty chunks → 0 coverage
        assert result["combined"] == pytest.approx(0.2, 0.01)

    def test_combined_weights(self):
        """Combined = relevance*0.5 + diversity*0.2 + coverage*0.3."""
        query = "Paris"
        chunks = [
            {"content": "Paris is beautiful", "relevance_score": 1.0},
            {"content": "Paris has nice food", "relevance_score": 1.0},
        ]
        result = quality_evaluator.evaluate_all(query, chunks)
        expected = (
            result["relevance"] * 0.5
            + result["diversity"] * 0.2
            + result["coverage"] * 0.3
        )
        assert result["combined"] == pytest.approx(expected, 0.001)


class TestChineseSupport:
    """Chinese text must use jieba tokenization instead of English word regex."""

    def test_chinese_diversity_not_degenerate(self):
        """Different Chinese chunks should have positive diversity (was always 0)."""
        chunks = [
            {"content": "李白是唐代著名的浪漫主义诗人，被称为诗仙。"},
            {"content": "杜甫是唐代现实主义诗人，其诗沉郁顿挫，被称为诗史。"},
        ]
        result = quality_evaluator.evaluate_diversity(chunks)
        assert 0.0 < result <= 1.0

    def test_chinese_identical_chunks_zero_diversity(self):
        chunks = [
            {"content": "李白是唐代诗人。"},
            {"content": "李白是唐代诗人。"},
        ]
        assert quality_evaluator.evaluate_diversity(chunks) == 0.0

    def test_chinese_coverage_partial(self):
        """Chinese query keywords should be matched against Chinese chunks."""
        result = quality_evaluator.evaluate_coverage(
            "李白的诗歌风格",
            [{"content": "李白的诗歌风格豪放飘逸，充满想象力。"}],
        )
        assert 0.0 < result <= 1.0

    def test_chinese_coverage_none(self):
        result = quality_evaluator.evaluate_coverage(
            "杜甫的诗歌",
            [{"content": "牛顿提出了万有引力定律"}],
        )
        assert result == 0.0
