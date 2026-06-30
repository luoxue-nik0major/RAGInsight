"""Tests for answer_evaluator (LLM-as-a-Judge)."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class MockEmbeddingFn:
    """Mock embedding function that returns simple random-like vectors."""
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        # Generate deterministic pseudo-embeddings based on text length
        result = []
        for i, t in enumerate(texts):
            np.random.seed(hash(t) % (2**31))
            vec = np.random.randn(512).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            result.append(vec)
        return result


@pytest.fixture
def emb_fn():
    return MockEmbeddingFn()


class TestFaithfulness:
    def test_all_claims_supported_with_relevant_context(self, emb_fn):
        from app.services.answer_evaluator import evaluate_faithfulness

        context = [
            {"content": "Paris is the capital of France.", "source": "doc1"},
            {"content": "France is a country in Western Europe.", "source": "doc2"},
        ]
        answer = "Paris is the capital of France. It is located in Western Europe."
        result = evaluate_faithfulness(answer, context, emb_fn)

        assert "score" in result
        assert 0 <= result["score"] <= 1
        assert result["total_claims"] > 0

    def test_empty_answer_returns_zero(self, emb_fn):
        from app.services.answer_evaluator import evaluate_faithfulness

        context = [{"content": "Some context.", "source": "doc1"}]
        result = evaluate_faithfulness("", context, emb_fn)

        assert result["score"] == 0.0
        assert result["total_claims"] == 0

    def test_empty_context_returns_zero(self, emb_fn):
        from app.services.answer_evaluator import evaluate_faithfulness

        result = evaluate_faithfulness("An answer sentence.", [], emb_fn)

        assert result["score"] == 0.0
        assert result["total_claims"] > 0
        assert result["supported_claims"] == 0

    def test_claim_results_have_required_fields(self, emb_fn):
        from app.services.answer_evaluator import evaluate_faithfulness

        context = [{"content": "Machine learning is a subset of artificial intelligence.", "source": "doc1"}]
        answer = "Machine learning is part of AI."
        result = evaluate_faithfulness(answer, context, emb_fn)

        for claim in result["claims"]:
            assert "text" in claim
            assert "supported" in claim
            assert "best_chunk_index" in claim
            assert "best_similarity" in claim

    def test_chinese_answer_faithfulness(self, emb_fn):
        from app.services.answer_evaluator import evaluate_faithfulness

        context = [
            {"content": "李白是唐代著名诗人，被誉为诗仙。", "source": "doc1"},
            {"content": "杜甫是唐代伟大的现实主义诗人。", "source": "doc2"},
        ]
        answer = "李白是唐代诗人。他被称作诗仙。"
        result = evaluate_faithfulness(answer, context, emb_fn)

        assert result["total_claims"] > 0


class TestRelevance:
    def test_similar_query_answer_high_relevance(self, emb_fn):
        from app.services.answer_evaluator import evaluate_relevance

        result = evaluate_relevance(
            "What is the capital of France?",
            "The capital of France is Paris.",
            emb_fn,
        )
        assert "score" in result
        assert 0 <= result["score"] <= 1
        assert result["method"] == "cosine_similarity"

    def test_empty_query_returns_zero(self, emb_fn):
        from app.services.answer_evaluator import evaluate_relevance

        result = evaluate_relevance("", "Some answer.", emb_fn)
        assert result["score"] == 0.0

    def test_empty_answer_returns_zero(self, emb_fn):
        from app.services.answer_evaluator import evaluate_relevance

        result = evaluate_relevance("Some query.", "", emb_fn)
        assert result["score"] == 0.0


class TestAnswerEvaluator:
    @patch("app.services.retriever._get_embedding_fn")
    def test_evaluate_returns_all_fields(self, mock_get_emb_fn, emb_fn):
        mock_get_emb_fn.return_value = emb_fn
        from app.services.answer_evaluator import answer_evaluator

        context = [
            {"content": "Paris is the capital of France.", "source": "doc1"},
            {"content": "France is in Western Europe.", "source": "doc2"},
        ]
        result = answer_evaluator.evaluate(
            query="What is the capital of France?",
            answer="Paris is the capital of France. It is in Europe.",
            context_chunks=context,
        )

        assert "faithfulness" in result
        assert "relevance" in result
        assert "combined_score" in result
        assert 0 <= result["combined_score"] <= 1
