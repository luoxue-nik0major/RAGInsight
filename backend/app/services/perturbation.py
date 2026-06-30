"""
Perturbation analyzer for RAG chunk importance scoring.

Uses a two-stage approach for cost optimization:
1. Approximate: embedding cosine similarity between chunk and original answer
2. Full LLM: leave-one-out perturbation for top-3 most important chunks
"""
import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from app.services.deepseek import deepseek_client


class TaskManager:
    """Simple in-memory task manager for perturbation jobs."""

    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._counter = 0
        self._lock = asyncio.Lock()

    async def create_task(self, session_id: int) -> str:
        async with self._lock:
            self._counter += 1
            task_id = f"perturb-{self._counter}"
            self._tasks[task_id] = {
                "task_id": task_id,
                "session_id": session_id,
                "status": "pending",
                "progress": 0,
                "total": 0,
                "result": None,
                "error": None,
                "created_at": time.time(),
            }
            return task_id

    def update_task(self, task_id: str, **kwargs):
        if task_id in self._tasks:
            # Merge extra_info if provided
            if "extra_info" in kwargs:
                extra = kwargs.pop("extra_info")
                current_extra = self._tasks[task_id].get("extra_info", {})
                current_extra.update(extra)
                kwargs["extra_info"] = current_extra
            self._tasks[task_id].update(kwargs)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._tasks.get(task_id)


task_manager = TaskManager()


class PerturbationAnalyzer:
    """Analyze chunk importance via leave-one-out perturbation."""

    def __init__(self, max_concurrent_llm: int = 2):
        self.semaphore = asyncio.Semaphore(max_concurrent_llm)
        self._embedding_fn: Optional[DefaultEmbeddingFunction] = None

    def _get_embedding_fn(self) -> DefaultEmbeddingFunction:
        if self._embedding_fn is None:
            self._embedding_fn = DefaultEmbeddingFunction()
        return self._embedding_fn

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts."""
        fn = self._get_embedding_fn()
        embeddings = fn(texts)
        # Ensure numpy array
        return np.array(embeddings, dtype=np.float32)

    def _cosine_sim(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two 1-D embeddings."""
        return float(cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0])

    async def analyze(
        self,
        query: str,
        original_answer: str,
        chunks: List[Dict[str, Any]],
        on_progress = None,
    ) -> List[Dict[str, Any]]:
        """
        Run full perturbation analysis.

        Returns list of dicts with keys:
        - chunk_index, chunk_id, content, importance_score, is_approximate, perturbed_answer
        """
        if not chunks:
            return []

        n = len(chunks)

        # Step 1: Approximate importance using chunk-to-answer embedding similarity
        chunk_texts = [c["content"] for c in chunks]
        chunk_embs = self._embed(chunk_texts)
        answer_emb = self._embed([original_answer])[0]

        approx_items = []
        for i in range(n):
            sim = self._cosine_sim(chunk_embs[i], answer_emb)
            approx_items.append({
                "index": i,
                "chunk_index": chunks[i]["chunk_index"],
                "chunk_id": chunks[i].get("id"),
                "content": chunks[i]["content"],
                "approximate_importance": float(sim),  # higher similarity -> more important
            })

        # Sort by approximate importance descending
        approx_items.sort(key=lambda x: x["approximate_importance"], reverse=True)

        # Determine top-3 for full LLM perturbation
        top3 = approx_items[:3]
        top3_original_indices = [item["index"] for item in top3]

        # Step 2: Full LLM perturbation for top-3 with concurrency control
        async def _perturb_one(original_idx: int) -> Dict[str, Any]:
            async with self.semaphore:
                removed = [c for j, c in enumerate(chunks) if j != original_idx]
                if not removed:
                    new_answer = "No context available."
                else:
                    new_answer = await deepseek_client.generate_answer(query, removed)

                # Compute embedding similarity between original and new answer
                orig_emb = self._embed([original_answer])[0]
                new_emb = self._embed([new_answer])[0]
                sim = self._cosine_sim(orig_emb, new_emb)
                importance = max(0.0, min(1.0, 1.0 - sim))

                return {
                    "original_index": original_idx,
                    "chunk_index": chunks[original_idx]["chunk_index"],
                    "importance_score": importance,
                    "perturbed_answer": new_answer,
                }

        tasks = [_perturb_one(idx) for idx in top3_original_indices]
        full_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build final result map
        result_map: Dict[int, Dict[str, Any]] = {}
        for item in approx_items:
            result_map[item["chunk_index"]] = {
                "chunk_index": item["chunk_index"],
                "chunk_id": item["chunk_id"],
                "content": item["content"],
                "importance_score": item["approximate_importance"],
                "is_approximate": True,
                "perturbed_answer": None,
            }

        for res in full_results:
            if isinstance(res, Exception):
                continue
            result_map[res["chunk_index"]]["importance_score"] = res["importance_score"]
            result_map[res["chunk_index"]]["is_approximate"] = False
            result_map[res["chunk_index"]]["perturbed_answer"] = res["perturbed_answer"]

        # Ensure all scores are normalized to [0, 1]
        final_results = list(result_map.values())
        for r in final_results:
            r["importance_score"] = max(0.0, min(1.0, float(r["importance_score"])))

        if on_progress:
            on_progress(n, n)

        return final_results

    async def what_if(
        self,
        query: str,
        original_answer: str,
        chunks: List[Dict[str, Any]],
        remove_indices: List[int],
    ) -> Dict[str, Any]:
        """
        Generate answer after removing specified chunks.
        Returns dict with new_answer and diff metrics.
        """
        kept = [c for i, c in enumerate(chunks) if c["chunk_index"] not in remove_indices]
        if not kept:
            new_answer = "No context available after removing selected chunks."
        else:
            new_answer = await deepseek_client.generate_answer(query, kept)

        orig_emb = self._embed([original_answer])[0]
        new_emb = self._embed([new_answer])[0]
        sim = self._cosine_sim(orig_emb, new_emb)

        return {
            "original_answer": original_answer,
            "new_answer": new_answer,
            "similarity": float(sim),
            "removed_count": len(remove_indices),
            "kept_count": len(kept),
        }
