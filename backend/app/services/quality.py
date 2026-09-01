"""
Quality evaluator for RAG retrieval results.
Evaluates from multiple dimensions: relevance, diversity, coverage.
"""
from typing import List, Dict, Any
import numpy as np
import re

from app.utils.text_utils import is_chinese_text, tokenize_for_bm25


class QualityEvaluator:
    """Evaluate retrieval result quality across multiple dimensions."""

    @staticmethod
    def evaluate_relevance(chunks: List[Dict[str, Any]]) -> float:
        """Average relevance score of retrieved chunks."""
        if not chunks:
            return 0.0
        scores = [c.get("relevance_score", 0.0) for c in chunks]
        return float(np.mean(scores))

    @staticmethod
    def evaluate_diversity(chunks: List[Dict[str, Any]]) -> float:
        """
        Measure semantic diversity of retrieved chunks.
        Uses simple text-based diversity: average pairwise Jaccard distance
        of word sets. Higher = more diverse.
        """
        if len(chunks) < 2:
            return 1.0  # Single result is trivially diverse

        def text_to_words(text: str) -> set:
            # Chinese: jieba tokenization; English: word regex, lowercase
            if is_chinese_text(text):
                return {w for w in tokenize_for_bm25(text) if w.strip()}
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return set(words)

        word_sets = [text_to_words(c.get("content", "")) for c in chunks]
        distances = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                union = word_sets[i] | word_sets[j]
                if union:
                    intersection = word_sets[i] & word_sets[j]
                    jaccard_dist = 1 - len(intersection) / len(union)
                    distances.append(jaccard_dist)
                # Both empty word sets carry no signal; skip the pair.

        return float(np.mean(distances)) if distances else 1.0

    @staticmethod
    def evaluate_coverage(query: str, chunks: List[Dict[str, Any]]) -> float:
        """
        Check coverage of query key entities/concepts in retrieved chunks.
        Extracts important words from query and checks presence in chunks.
        """
        if not chunks:
            return 0.0

        # Extract potential entities/keywords from query
        # Words with 4+ chars, excluding common stop words
        stop_words = {
            "what", "which", "where", "when", "why", "how", "who", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "do",
            "does", "did", "will", "would", "could", "should", "may", "might",
            "can", "the", "a", "an", "this", "that", "these", "those", "and",
            "or", "but", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "from",
            "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        }

        if is_chinese_text(query):
            # Chinese: jieba tokens with 2+ characters
            query_words = {
                w.strip() for w in tokenize_for_bm25(query) if len(w.strip()) >= 2
            }
        else:
            query_words = set(
                w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', query)
                if w.lower() not in stop_words
            )

        if not query_words:
            return 1.0

        chunk_text = " ".join(c.get("content", "").lower() for c in chunks)
        covered = sum(1 for w in query_words if w in chunk_text)
        return covered / len(query_words)

    @staticmethod
    def evaluate_all(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Run all quality evaluations and return combined scores."""
        relevance = QualityEvaluator.evaluate_relevance(chunks)
        diversity = QualityEvaluator.evaluate_diversity(chunks)
        coverage = QualityEvaluator.evaluate_coverage(query, chunks)

        # Combined quality score: weighted average
        combined = relevance * 0.5 + diversity * 0.2 + coverage * 0.3

        return {
            "relevance": round(relevance, 4),
            "diversity": round(diversity, 4),
            "coverage": round(coverage, 4),
            "combined": round(combined, 4),
        }


quality_evaluator = QualityEvaluator()
