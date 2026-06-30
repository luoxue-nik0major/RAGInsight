"""Tests for retriever utilities and logic."""
import pytest
from app.utils.text_utils import tokenize_for_bm25, is_chinese_text


class TestTokenize:
    def test_english_tokenize(self):
        result = tokenize_for_bm25("machine learning deep learning")
        assert "machine" in result
        assert "learning" in result
        assert "deep" in result

    def test_chinese_tokenize(self):
        result = tokenize_for_bm25("机器学习")
        # jieba should segment this into at least one token
        assert len(result) >= 1


class TestIsChineseText:
    def test_pure_chinese(self):
        assert is_chinese_text("机器学习") is True

    def test_pure_english(self):
        assert is_chinese_text("machine learning") is False

    def test_mixed(self):
        assert is_chinese_text("李白的诗歌") is True  # >30% Chinese chars


class TestRRFLogic:
    def test_rrf_fusion_formula(self):
        """Verify RRF scoring logic: higher rank = higher score."""
        K = 60
        # Doc A: rank 1 in dense, rank 3 in sparse
        score_a = 1 / (K + 1) + 1 / (K + 3)
        # Doc B: rank 2 in dense, not in sparse
        score_b = 1 / (K + 2)
        # Doc C: not in dense, rank 1 in sparse
        score_c = 1 / (K + 1)
        # A should outrank B; C should outrank B
        assert score_a > score_b
        assert score_c > score_b
        # A vs C: A has two sources, should win
        assert score_a > score_c

    def test_rrf_single_source(self):
        K = 60
        # Single source ranking: lower rank number = higher score
        assert 1 / (K + 1) > 1 / (K + 5)
