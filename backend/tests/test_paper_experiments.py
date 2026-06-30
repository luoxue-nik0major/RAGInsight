"""
Integration tests for paper experiments (E1-E5).

Validates that all 5 paper experiments produce valid output structures.
These tests verify the metric computation pipeline without running actual experiments.
"""
import pytest
import json
import os
import sys


# ── Sample Data ─────────────────────────────────────────────────────────

def make_sample_experiment_results(n_queries: int = 10) -> list:
    """Generate synthetic experiment results matching the real output schema."""
    results = []
    strategies = ["vector", "hybrid"]
    alert_types = ["empty_results", "low_relevance", "low_coverage", "low_diversity"]

    for i in range(n_queries):
        for strategy in strategies:
            results.append({
                "query_id": 100 + i,
                "query": f"Test query {i}",
                "strategy": strategy,
                "category": "simple_fact" if i < 5 else "multi_hop",
                "expected_alerts": alert_types[:i % 3],
                "session_id": 1000 + i * 10 + strategies.index(strategy),
                "status": "completed",
                "complexity_score": 0.2 + i * 0.05,
                "recommended_strategy": "vector" if i < 3 else "hybrid",
                "final_answer": f"Answer for query {i} with {strategy}",
                "actual_alerts": alert_types[:i % 3 + 1],
                "steps": [
                    {
                        "step_type": "vector_retrieve",
                        "input_data": {"strategy": strategy, "top_k": 5},
                        "output_data": {
                            "quality_metrics": {
                                "relevance": 0.5 + 0.03 * i,
                                "coverage": 0.4 + 0.02 * i,
                                "diversity": 0.3 + 0.04 * i,
                            }
                        },
                        "quality_score": 0.6,
                        "duration_ms": 100 + i * 20,
                        "chunks": [],
                        "alerts": [],
                    },
                    {
                        "step_type": "answer_generate",
                        "input_data": {},
                        "output_data": {"answer": f"Answer {i}"},
                        "quality_score": 0.7,
                        "duration_ms": 500 + i * 50,
                        "chunks": [],
                        "alerts": [],
                    },
                ],
                "execution_trace": {
                    "answer_evaluation": {
                        "faithfulness": {"score": 0.6 + 0.02 * i},
                        "relevance": {"score": 0.55 + 0.02 * i},
                        "combined_score": 0.58 + 0.02 * i,
                    }
                },
            })
    return results


# ── Test Experiments 1 & 2 ──────────────────────────────────────────────

class TestFaultDetectionMetrics:
    def test_compute_fault_detection(self):
        from app.routers.experiments import _compute_fault_detection_metrics

        results = make_sample_experiment_results(10)
        metrics = _compute_fault_detection_metrics(results)

        assert "per_alert_type" in metrics
        assert "macro" in metrics
        assert "micro" in metrics
        assert "precision" in metrics["macro"]
        assert "recall" in metrics["macro"]
        assert "f1" in metrics["macro"]
        assert 0.0 <= metrics["macro"]["f1"] <= 1.0

    def test_fault_detection_empty(self):
        from app.routers.experiments import _compute_fault_detection_metrics

        metrics = _compute_fault_detection_metrics([])
        assert metrics["macro"]["f1"] == 0.0
        assert metrics["per_alert_type"] == {}

    def test_compute_strategy_comparison(self):
        from app.routers.experiments import _compute_strategy_comparison

        results = make_sample_experiment_results(10)
        comparison = _compute_strategy_comparison(results)

        assert "strategies" in comparison
        assert "vector" in comparison["strategies"]
        assert "hybrid" in comparison["strategies"]

        for strategy, stats in comparison["strategies"].items():
            assert "avg_relevance" in stats
            assert "avg_coverage" in stats
            assert "avg_diversity" in stats
            assert "avg_answer_quality" in stats
            assert "avg_faithfulness" in stats
            assert "avg_answer_relevance" in stats
            assert "avg_duration_ms" in stats
            assert stats["count"] > 0
            assert 0.0 <= stats["avg_faithfulness"] <= 1.0

    def test_strategy_comparison_empty(self):
        from app.routers.experiments import _compute_strategy_comparison

        comparison = _compute_strategy_comparison([])
        assert comparison == {"strategies": {}}


# ── Test Experiment 5: Router Comparison ────────────────────────────────

class TestRouterComparison:
    def test_router_trainer_heuristic_accuracy_empty(self):
        from app.services.router_trainer import RouterTrainer
        trainer = RouterTrainer()
        stats = trainer.compute_heuristic_accuracy()
        assert stats == {"accuracy": 0.0, "correct": 0, "total": 0}

    def test_router_trainer_with_data(self):
        from app.services.router_trainer import RouterTrainer
        trainer = RouterTrainer()
        trainer._training_data = [
            {"oracle_strategy": "vector", "heuristic_strategy": "vector"},
            {"oracle_strategy": "hybrid", "heuristic_strategy": "vector"},
            {"oracle_strategy": "vector", "heuristic_strategy": "vector"},
        ]
        stats = trainer.compute_heuristic_accuracy()
        assert abs(stats["accuracy"] - 2/3) < 0.001
        assert stats["total"] == 3

    def test_strategy_distribution(self):
        from app.services.router_trainer import RouterTrainer
        trainer = RouterTrainer()
        trainer._training_data = [
            {"oracle_strategy": "vector"},
            {"oracle_strategy": "vector"},
            {"oracle_strategy": "hybrid"},
            {"oracle_strategy": "graph"},
        ]
        dist = trainer.get_strategy_distribution()
        assert dist == {"vector": 2, "hybrid": 1, "graph": 1}

    def test_router_interpretation_positive(self):
        from app.routers.experiments import _interpret_router_results
        result = _interpret_router_results({
            "learned_accuracy": 0.75, "heuristic_accuracy": 0.62, "improvement": 0.13
        })
        assert "显著优于" in result

    def test_router_interpretation_negative(self):
        from app.routers.experiments import _interpret_router_results
        result = _interpret_router_results({
            "learned_accuracy": 0.55, "heuristic_accuracy": 0.62, "improvement": -0.07
        })
        assert "不如" in result or "不如启发式" in result


# ── Test Paper Exporter ─────────────────────────────────────────────────

class TestPaperExporter:
    def test_generate_fault_detection_table(self):
        from app.services.paper_exporter import PaperExporter
        exporter = PaperExporter()

        # Use a temp dir
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.output_dir = tmpdir
            metrics = {
                "per_alert_type": {
                    "empty_results": {"precision": 0.9, "recall": 0.95, "f1": 0.92, "tp": 10, "fp": 1, "fn": 1},
                    "low_relevance": {"precision": 0.7, "recall": 0.65, "f1": 0.67, "tp": 7, "fp": 3, "fn": 4},
                },
                "macro": {"precision": 0.80, "recall": 0.80, "f1": 0.80},
                "micro": {"precision": 0.82, "recall": 0.82, "f1": 0.82},
            }
            latex = exporter.generate_fault_detection_table(metrics)
            assert "empty\\_results" in latex
            assert "Macro Avg" in latex
            assert "tabular" in latex

    def test_generate_strategy_comparison_table(self):
        from app.services.paper_exporter import PaperExporter
        exporter = PaperExporter()

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.output_dir = tmpdir
            metrics = {
                "strategies": {
                    "vector": {"count": 10, "avg_relevance": 0.65, "avg_coverage": 0.55, "avg_diversity": 0.45, "avg_answer_quality": 0.72, "avg_faithfulness": 0.68, "avg_answer_relevance": 0.60, "avg_duration_ms": 450},
                    "hybrid": {"count": 10, "avg_relevance": 0.71, "avg_coverage": 0.62, "avg_diversity": 0.52, "avg_answer_quality": 0.75, "avg_faithfulness": 0.73, "avg_answer_relevance": 0.64, "avg_duration_ms": 580},
                }
            }
            latex = exporter.generate_strategy_comparison_table(metrics)
            assert "vector" in latex
            assert "hybrid" in latex

    def test_export_all_empty(self):
        from app.services.paper_exporter import PaperExporter
        exporter = PaperExporter()

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.output_dir = tmpdir
            outputs = exporter.export_all({})
            assert "output_dir" in outputs
            assert len(outputs.get("files_generated", [])) == 0


# ── Test Answer Evaluation Extensions ───────────────────────────────────

class TestExtendedEvaluation:
    def test_evaluate_context_precision(self):
        from app.services.answer_evaluator import evaluate_context_precision

        chunks = [
            {"content": "Chunk 0 content", "chunk_index": 0},
            {"content": "Chunk 1 content", "chunk_index": 1},
            {"content": "Chunk 2 content", "chunk_index": 2},
        ]
        result = evaluate_context_precision(
            "Answer with [ref:chunk_0] and [ref:chunk_1].",
            chunks,
        )
        assert abs(result["score"] - 2/3) < 0.001
        assert result["cited_count"] == 2
        assert result["total_chunks"] == 3

    def test_evaluate_context_precision_no_citations(self):
        from app.services.answer_evaluator import evaluate_context_precision

        chunks = [{"content": "Test", "chunk_index": 0}]
        result = evaluate_context_precision("Answer without citations.", chunks)
        assert result["score"] == 0.0
        assert result["cited_count"] == 0

    def test_evaluate_context_precision_empty_chunks(self):
        from app.services.answer_evaluator import evaluate_context_precision

        result = evaluate_context_precision("Answer.", [])
        assert result["score"] == 0.0

    def test_evaluate_exact_match(self):
        from app.services.answer_evaluator import evaluate_exact_match

        result = evaluate_exact_match(
            "The capital of France is Paris.",
            ["Paris is the capital of France.", "The capital is Paris."],
        )
        assert 0.0 <= result["score"] <= 1.0
        assert "f1" in result

    def test_evaluate_exact_match_no_references(self):
        from app.services.answer_evaluator import evaluate_exact_match

        result = evaluate_exact_match("Some answer.", [])
        assert result["score"] == 0.0
