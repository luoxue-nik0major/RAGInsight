"""
Training data collector for learned query-to-strategy router.

Runs all queries across all 3 strategies (full grid) and exports
labeled training data for the strategy classifier.
"""
import json
import os
import asyncio
from typing import List, Dict, Any, Optional

from app.services.complexity import complexity_analyzer
from app.services.experiment_runner import experiment_runner
from app.utils.text_utils import is_chinese_text

TRAINING_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments",
)


class RouterTrainer:
    """Collects oracle-labeled training data for strategy routing."""

    def __init__(self):
        self._training_data: List[Dict[str, Any]] = []

    async def collect_training_data(
        self,
        queries: List[Dict[str, Any]],
        strategies: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run full-grid experiment and compute oracle labels.

        For each query, runs all strategies, records features + quality,
        and labels the best-performing strategy as the oracle choice.
        """
        strategies = strategies or ["vector", "hybrid", "graph"]

        # Run full grid experiment
        results = await experiment_runner.run_batch(queries, strategies)

        # Group results by query_id
        by_query: Dict[int, List[Dict[str, Any]]] = {}
        for r in results:
            qid = r.get("query_id", hash(r["query"]) % 100000)
            if qid not in by_query:
                by_query[qid] = []
            by_query[qid].append(r)

        training_samples = []

        for qid, query_results in by_query.items():
            if not query_results:
                continue

            query_text = query_results[0]["query"]
            is_chinese = is_chinese_text(query_text)

            # Extract complexity features
            complexity = complexity_analyzer.analyze(query_text)
            features = complexity["features"]

            # Determine oracle strategy (highest answer quality)
            best_strategy = None
            best_quality = -1.0

            per_strategy_qualities = {}

            for r in query_results:
                strategy = r["strategy"]
                quality = 0.0

                trace = r.get("execution_trace", {}) or {}
                answer_eval = trace.get("answer_evaluation")
                if answer_eval and isinstance(answer_eval, dict):
                    quality = answer_eval.get("combined_score", 0.0)
                elif r.get("status") == "failed":
                    quality = 0.0
                else:
                    # Fallback: use retrieval quality combined score
                    for step in r.get("steps", []):
                        if step.get("step_type") == "vector_retrieve":
                            qs = step.get("quality_score")
                            if qs is not None:
                                quality = max(quality, qs)

                per_strategy_qualities[strategy] = round(quality, 4)

                if quality > best_quality:
                    best_quality = quality
                    best_strategy = strategy

            if best_strategy is None:
                best_strategy = query_results[0]["strategy"]

            # Build training sample
            sample = {
                "query_id": qid,
                "query": query_text,
                "language": "zh" if is_chinese else "en",
                "complexity_score": complexity["complexity_score"],
                "question_type": complexity["question_type"],
                "features": {
                    "length": features["length"],
                    "length_score": features["length_score"],
                    "sentence_count": features["sentence_count"],
                    "sentence_score": features["sentence_score"],
                    "entity_count": features["entity_count"],
                    "entity_score": features["entity_score"],
                    "relation_count": features["relation_count"],
                    "relation_score": features["relation_score"],
                    "semantic_score": features["semantic_score"],
                    "hop_demand_score": features["hop_demand_score"],
                },
                "per_strategy_quality": per_strategy_qualities,
                "oracle_strategy": best_strategy,
                "oracle_quality": round(best_quality, 4),
                "heuristic_strategy": self._get_heuristic_prediction(complexity["complexity_score"]),
            }
            training_samples.append(sample)

        self._training_data = training_samples

        # Save to disk
        os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
        filepath = os.path.join(TRAINING_DATA_DIR, "router_training_data.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(training_samples, f, ensure_ascii=False, indent=2)

        return training_samples

    def _get_heuristic_prediction(self, complexity_score: float) -> str:
        if complexity_score < 0.30:
            return "vector"
        elif complexity_score < 0.70:
            return "hybrid"
        else:
            return "graph"

    def get_training_data(self) -> List[Dict[str, Any]]:
        return self._training_data.copy()

    def load_training_data(self) -> Optional[List[Dict[str, Any]]]:
        """Load previously saved training data from disk."""
        filepath = os.path.join(TRAINING_DATA_DIR, "router_training_data.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._training_data = data
        return data

    def compute_heuristic_accuracy(self) -> Dict[str, Any]:
        """Compute how often the heuristic router matches the oracle."""
        if not self._training_data:
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        correct = sum(
            1 for s in self._training_data
            if s["heuristic_strategy"] == s["oracle_strategy"]
        )
        total = len(self._training_data)
        return {
            "accuracy": round(correct / total, 4) if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        }

    def get_strategy_distribution(self) -> Dict[str, int]:
        """Get oracle strategy distribution in training data."""
        dist = {"vector": 0, "hybrid": 0, "graph": 0}
        for s in self._training_data:
            oracle = s["oracle_strategy"]
            if oracle in dist:
                dist[oracle] += 1
        return dist


router_trainer = RouterTrainer()
