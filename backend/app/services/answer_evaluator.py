"""
LLM-as-a-Judge answer evaluator using local embedding models.
Computes answer faithfulness and relevance without calling external APIs.
"""
import re
import numpy as np
from typing import List, Dict, Any


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences for both Chinese and English."""
    sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))


def evaluate_faithfulness(
    answer: str,
    context_chunks: List[Dict[str, Any]],
    embedding_fn,
) -> Dict[str, Any]:
    """
    Evaluate whether each claim in the answer is supported by the retrieved context.

    Returns:
        dict with:
        - score: overall faithfulness (0-1)
        - claims: list of {text, supported, best_chunk_index, best_similarity}
    """
    claims = _split_sentences(answer)
    if not claims:
        return {"score": 0.0, "claims": [], "total_claims": 0, "supported_claims": 0}

    # Embed context chunks (cached by the embedding function internally)
    chunk_texts = [c["content"] for c in context_chunks]
    if not chunk_texts:
        return {"score": 0.0, "claims": [], "total_claims": len(claims), "supported_claims": 0}

    try:
        chunk_embeddings = embedding_fn(chunk_texts)
        claim_embeddings = embedding_fn(claims)

        supported_count = 0
        claim_results = []
        similarity_threshold = 0.45

        for i, claim in enumerate(claims):
            best_sim = 0.0
            best_chunk = -1
            for j in range(len(chunk_texts)):
                sim = _cosine_similarity(claim_embeddings[i], chunk_embeddings[j])
                if sim > best_sim:
                    best_sim = sim
                    best_chunk = j

            supported = best_sim >= similarity_threshold
            if supported:
                supported_count += 1

            claim_results.append({
                "text": claim,
                "supported": supported,
                "best_chunk_index": best_chunk,
                "best_similarity": round(float(best_sim), 4),
            })

        score = supported_count / len(claims)
        return {
            "score": round(score, 4),
            "claims": claim_results,
            "total_claims": len(claims),
            "supported_claims": supported_count,
        }
    except Exception as e:
        return {"score": 0.0, "claims": [], "total_claims": len(claims), "supported_claims": 0, "error": str(e)}


def evaluate_relevance(
    query: str,
    answer: str,
    embedding_fn,
) -> Dict[str, Any]:
    """
    Evaluate answer relevance by computing semantic similarity between query and answer.

    Returns:
        dict with score (0-1) — how semantically close the answer is to the query.
    """
    if not answer or not query:
        return {"score": 0.0, "method": "cosine_similarity"}

    try:
        embeddings = embedding_fn([query, answer])
        sim = _cosine_similarity(embeddings[0], embeddings[1])
        # Clamp to [0, 1] — cosine can sometimes be slightly negative for unrelated texts
        score = max(0.0, min(1.0, sim))
        return {"score": round(score, 4), "method": "cosine_similarity"}
    except Exception as e:
        return {"score": 0.0, "method": "cosine_similarity", "error": str(e)}


class AnswerEvaluator:
    """Evaluate answer quality using local embedding models."""

    def evaluate(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run both faithfulness and relevance evaluations.
        """
        from app.services.retriever import _get_embedding_fn
        emb_fn = _get_embedding_fn()

        faithfulness = evaluate_faithfulness(answer, context_chunks, emb_fn)
        relevance = evaluate_relevance(query, answer, emb_fn)

        combined = round(
            faithfulness["score"] * 0.6 + relevance["score"] * 0.4,
            4,
        )

        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "combined_score": combined,
        }


def evaluate_context_precision(
    answer: str,
    context_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate the proportion of retrieved chunks that were actually cited by the LLM.
    Context precision = cited_chunks / total_retrieved_chunks
    """
    if not context_chunks:
        return {"score": 0.0, "cited_count": 0, "total_chunks": 0}

    import re
    cited_indices = set()
    for match in re.finditer(r'\[ref:chunk_(\d+)\]', answer):
        cited_indices.add(int(match.group(1)))

    valid_chunk_indices = set(c.get("chunk_index", i) for i, c in enumerate(context_chunks))
    valid_citations = cited_indices & valid_chunk_indices

    score = len(valid_citations) / len(context_chunks) if context_chunks else 0.0
    return {
        "score": round(min(score, 1.0), 4),
        "cited_count": len(valid_citations),
        "total_chunks": len(context_chunks),
    }


def evaluate_context_recall(
    answer: str,
    query: str,
    context_chunks: List[Dict[str, Any]],
    embedding_fn,
) -> Dict[str, Any]:
    """
    Approximate context recall: what fraction of answer content is covered by chunks.
    Uses embedding similarity to estimate information coverage.
    """
    if not answer or not context_chunks:
        return {"score": 0.0, "method": "embedding_coverage"}

    try:
        claims = _split_sentences(answer)
        if not claims:
            return {"score": 0.0, "method": "embedding_coverage"}

        chunk_texts = [c["content"] for c in context_chunks]
        chunk_embs = embedding_fn(chunk_texts)
        claim_embs = embedding_fn(claims)

        covered = 0
        threshold = 0.35  # lower threshold than faithfulness (0.45)
        for claim_emb in claim_embs:
            best_sim = max(
                _cosine_similarity(claim_emb, chunk_emb)
                for chunk_emb in chunk_embs
            )
            if best_sim >= threshold:
                covered += 1

        score = covered / len(claims)
        return {
            "score": round(score, 4),
            "covered_claims": covered,
            "total_claims": len(claims),
            "method": "embedding_coverage",
        }
    except Exception as e:
        return {"score": 0.0, "method": "embedding_coverage", "error": str(e)}


def evaluate_exact_match(
    answer: str,
    references: List[str],
) -> Dict[str, Any]:
    """
    Compute token-level F1 against reference answers.
    """
    if not answer or not references:
        return {"score": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    import re

    def tokenize(text: str) -> set:
        # Simple whitespace + punctuation tokenization
        tokens = set(re.findall(r'\b\w+\b', text.lower()))
        return tokens

    answer_tokens = tokenize(answer)

    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for ref in references:
        ref_tokens = tokenize(ref)
        if not ref_tokens:
            continue

        intersection = answer_tokens & ref_tokens
        precision = len(intersection) / len(answer_tokens) if answer_tokens else 0.0
        recall = len(intersection) / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall

    return {
        "score": round(best_f1, 4),
        "f1": round(best_f1, 4),
        "precision": round(best_precision, 4),
        "recall": round(best_recall, 4),
    }


answer_evaluator = AnswerEvaluator()
