"""
Experiment API routes for Phase 5.
"""
import asyncio
import json
import os
from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool
from typing import List, Dict, Any

from app.services.experiment_runner import experiment_runner, EXPERIMENT_DIR
from app.services.export_service import export_session_json
from app.services.router_trainer import router_trainer
from app.services.strategy_classifier import strategy_classifier

router = APIRouter(prefix="/api", tags=["experiments"])

DATASETS = {
    "squad": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "squad_test_queries.json"),
    "chinese": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chinese_test_queries.json"),
}


def _load_dataset(dataset_name: str = "squad") -> List[Dict[str, Any]]:
    path = DATASETS.get(dataset_name, DATASETS["squad"])
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", [])


def _read_latest_jsonl() -> List[Dict[str, Any]]:
    """Read the newest experiment result file (blocking; call via run_in_threadpool)."""
    files = sorted(
        [f for f in os.listdir(EXPERIMENT_DIR) if f.endswith(".jsonl")],
        reverse=True,
    )
    if not files:
        return []
    with open(os.path.join(EXPERIMENT_DIR, files[0]), "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _compute_fault_detection_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Experiment 1: Fault detection accuracy.
    Compares expected_alerts (human labels) vs actual_alerts (system detection).
    """
    # Collect all alert types
    all_alert_types = set()
    for r in results:
        all_alert_types.update(r.get("expected_alerts", []))
        all_alert_types.update(r.get("actual_alerts", []))

    per_type = {}
    global_tp = global_fp = global_fn = 0

    for alert_type in sorted(all_alert_types):
        tp = fp = fn = 0
        for r in results:
            expected = set(r.get("expected_alerts", []))
            actual = set(r.get("actual_alerts", []))

            # For this alert type
            if alert_type in expected and alert_type in actual:
                tp += 1
            elif alert_type not in expected and alert_type in actual:
                fp += 1
            elif alert_type in expected and alert_type not in actual:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_type[alert_type] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        global_tp += tp
        global_fp += fp
        global_fn += fn

    macro_precision = sum(v["precision"] for v in per_type.values()) / len(per_type) if per_type else 0.0
    macro_recall = sum(v["recall"] for v in per_type.values()) / len(per_type) if per_type else 0.0
    macro_f1 = sum(v["f1"] for v in per_type.values()) / len(per_type) if per_type else 0.0

    global_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    global_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0

    return {
        "per_alert_type": per_type,
        "macro": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
        },
        "micro": {
            "precision": round(global_precision, 4),
            "recall": round(global_recall, 4),
            "f1": round(global_f1, 4),
            "tp": global_tp,
            "fp": global_fp,
            "fn": global_fn,
        },
    }


def _compute_strategy_comparison(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Experiment 2: Cross-strategy comparison (vector vs hybrid).
    Includes answer faithfulness and relevance from LLM-as-a-Judge evaluation.
    """
    strategy_stats: Dict[str, Dict[str, List[float]]] = {}

    for r in results:
        strategy = r.get("strategy", "unknown")
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {
                "relevance": [],
                "coverage": [],
                "diversity": [],
                "answer_quality": [],
                "faithfulness": [],
                "answer_relevance": [],
                "duration_ms": [],
            }

        # Extract quality metrics from retrieve step
        for step in r.get("steps", []):
            if step.get("step_type") == "vector_retrieve":
                qm = step.get("output_data", {}).get("quality_metrics", {})
                strategy_stats[strategy]["relevance"].append(qm.get("relevance", 0))
                strategy_stats[strategy]["coverage"].append(qm.get("coverage", 0))
                strategy_stats[strategy]["diversity"].append(qm.get("diversity", 0))
            elif step.get("step_type") == "answer_generate":
                strategy_stats[strategy]["answer_quality"].append(step.get("quality_score", 0) or 0)
                strategy_stats[strategy]["duration_ms"].append(step.get("duration_ms", 0) or 0)

        # Extract answer evaluation from execution_trace
        answer_eval = r.get("execution_trace", {}).get("answer_evaluation")
        if answer_eval:
            strategy_stats[strategy]["faithfulness"].append(answer_eval.get("faithfulness", {}).get("score", 0))
            strategy_stats[strategy]["answer_relevance"].append(answer_eval.get("relevance", {}).get("score", 0))
        else:
            strategy_stats[strategy]["faithfulness"].append(0)
            strategy_stats[strategy]["answer_relevance"].append(0)

    summary = {}
    for strategy, stats in strategy_stats.items():
        summary[strategy] = {
            "count": len(stats["relevance"]),
            "avg_relevance": round(sum(stats["relevance"]) / len(stats["relevance"]), 4) if stats["relevance"] else 0,
            "avg_coverage": round(sum(stats["coverage"]) / len(stats["coverage"]), 4) if stats["coverage"] else 0,
            "avg_diversity": round(sum(stats["diversity"]) / len(stats["diversity"]), 4) if stats["diversity"] else 0,
            "avg_answer_quality": round(sum(stats["answer_quality"]) / len(stats["answer_quality"]), 4) if stats["answer_quality"] else 0,
            "avg_faithfulness": round(sum(stats["faithfulness"]) / len(stats["faithfulness"]), 4) if stats["faithfulness"] else 0,
            "avg_answer_relevance": round(sum(stats["answer_relevance"]) / len(stats["answer_relevance"]), 4) if stats["answer_relevance"] else 0,
            "avg_duration_ms": round(sum(stats["duration_ms"]) / len(stats["duration_ms"]), 4) if stats["duration_ms"] else 0,
        }

    return {"strategies": summary}


@router.post("/experiments/run")
async def run_experiments(dataset: str = "squad"):
    """Trigger batch experiment run."""
    if experiment_runner.is_running():
        raise HTTPException(status_code=409, detail="Experiment already running")

    queries = _load_dataset(dataset)
    if not queries:
        raise HTTPException(status_code=400, detail="No test queries found")

    # Run in background
    asyncio.create_task(experiment_runner.run_batch(queries, strategies=["vector", "hybrid"]))

    return {"status": "started", "total_queries": len(queries), "strategies": ["vector", "hybrid"], "dataset": dataset}


@router.get("/experiments/status")
async def get_experiment_status():
    """Get current experiment progress."""
    progress = experiment_runner.get_progress()
    return {
        **progress,
        "is_running": experiment_runner.is_running(),
    }


@router.get("/experiments/results")
async def get_experiment_results():
    """Get latest experiment results."""
    results = experiment_runner.get_latest_results()
    return {"results": results, "count": len(results)}


@router.get("/experiments/metrics")
async def get_experiment_metrics():
    """Get experiment 1 & 2 metrics."""
    results = experiment_runner.get_latest_results()
    if not results:
        # Try loading from latest file
        latest = await run_in_threadpool(_read_latest_jsonl)
        if latest:
            results = latest

    if not results:
        return {
            "experiment_1_fault_detection": None,
            "experiment_2_strategy_comparison": None,
            "message": "No experiment results available. Run experiments first.",
        }

    return {
        "experiment_1_fault_detection": _compute_fault_detection_metrics(results),
        "experiment_2_strategy_comparison": _compute_strategy_comparison(results),
        "total_runs": len(results),
    }


@router.get("/experiments/dataset")
async def get_experiment_dataset(dataset: str = "squad"):
    """Get test query dataset."""
    queries = _load_dataset(dataset)
    return {"queries": queries, "count": len(queries), "dataset": dataset}


@router.post("/experiments/export/{session_id}")
async def export_experiment_session(session_id: int):
    """Export a single session as JSON (Experiment 3)."""
    data = await export_session_json(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": data}


def _generate_markdown_report(metrics: Dict[str, Any]) -> str:
    """Generate a Markdown report from experiment metrics."""
    lines = []
    lines.append("# RAGInsight 实验报告")
    lines.append("")
    lines.append(f"**总运行次数**: {metrics.get('total_runs', 0)}")
    lines.append("")

    # Experiment 1
    exp1 = metrics.get("experiment_1_fault_detection")
    if exp1:
        lines.append("## 实验一：故障检测准确率")
        lines.append("")
        macro = exp1.get("macro", {})
        lines.append(f"- **Macro Precision**: {macro.get('precision', 0):.2%}")
        lines.append(f"- **Macro Recall**: {macro.get('recall', 0):.2%}")
        lines.append(f"- **Macro F1**: {macro.get('f1', 0):.2%}")
        lines.append("")
        lines.append("| Alert Type | Precision | Recall | F1 | TP | FP | FN |")
        lines.append("|------------|-----------|--------|----|----|----|----|")
        for alert_type, m in exp1.get("per_alert_type", {}).items():
            lines.append(f"| {alert_type} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} | {m['tp']} | {m['fp']} | {m['fn']} |")
        lines.append("")

    # Experiment 2
    exp2 = metrics.get("experiment_2_strategy_comparison")
    if exp2:
        lines.append("## 实验二：跨检索架构对比")
        lines.append("")
        for strategy, stats in exp2.get("strategies", {}).items():
            lines.append(f"### {strategy}")
            lines.append(f"- 运行次数: {stats['count']}")
            lines.append(f"- 平均相关性: {stats['avg_relevance']:.2%}")
            lines.append(f"- 平均覆盖率: {stats['avg_coverage']:.2%}")
            lines.append(f"- 平均多样性: {stats['avg_diversity']:.2%}")
            lines.append(f"- 平均答案质量: {stats['avg_answer_quality']:.2%}")
            lines.append(f"- 平均答案忠实度: {stats['avg_faithfulness']:.2%}")
            lines.append(f"- 平均答案相关性: {stats['avg_answer_relevance']:.2%}")
            lines.append(f"- 平均耗时: {stats['avg_duration_ms']:.0f}ms")
            lines.append("")

    return "\n".join(lines)


@router.get("/experiments/report")
async def get_experiment_report():
    """Get experiment report as Markdown."""
    results = experiment_runner.get_latest_results()
    if not results:
        latest = await run_in_threadpool(_read_latest_jsonl)
        if latest:
            results = latest

    if not results:
        raise HTTPException(status_code=404, detail="No experiment results available")

    metrics = {
        "experiment_1_fault_detection": _compute_fault_detection_metrics(results),
        "experiment_2_strategy_comparison": _compute_strategy_comparison(results),
        "total_runs": len(results),
    }
    report_md = _generate_markdown_report(metrics)
    return {"report": report_md, "format": "markdown"}


# ── Full-Grid Experiment & Router Training ──────────────────────────────

@router.post("/experiments/full-grid")
async def run_full_grid_experiment(dataset: str = "squad"):
    """Run all queries across all 3 strategies (full grid) for router training."""
    if experiment_runner.is_running():
        raise HTTPException(status_code=409, detail="Experiment already running")

    queries = _load_dataset(dataset)
    if not queries:
        raise HTTPException(status_code=400, detail="No test queries found")

    asyncio.create_task(
        experiment_runner.run_batch(queries, strategies=["vector", "hybrid", "graph"])
    )

    return {
        "status": "started",
        "total_queries": len(queries),
        "strategies": ["vector", "hybrid", "graph"],
        "dataset": dataset,
        "total_runs": len(queries) * 3,
    }


@router.get("/experiments/router-training-data")
async def get_router_training_data():
    """Export oracle-labeled training data for strategy classifier."""
    # Try loading from disk first
    data = router_trainer.load_training_data()
    if data is None:
        # Generate from latest experiment results
        results = experiment_runner.get_latest_results()
        if not results:
            results = await run_in_threadpool(_read_latest_jsonl)

        if not results:
            raise HTTPException(status_code=404, detail="No experiment results available. Run full-grid first.")

        # Convert results to training data format
        queries = _load_dataset()
        data = await router_trainer.collect_training_data(queries, ["vector", "hybrid", "graph"])

    heuristic_stats = router_trainer.compute_heuristic_accuracy()

    return {
        "samples": data,
        "count": len(data),
        "heuristic_accuracy": heuristic_stats["accuracy"],
        "heuristic_correct": heuristic_stats["correct"],
        "strategy_distribution": router_trainer.get_strategy_distribution(),
    }


@router.post("/experiments/train-router")
async def train_strategy_router():
    """Train the learned strategy classifier on existing training data."""
    data = router_trainer.load_training_data()
    if data is None:
        raise HTTPException(status_code=404, detail="No training data. Run full-grid and export training data first.")

    result = strategy_classifier.train(data)
    return result


@router.get("/experiments/router-comparison")
async def get_router_comparison():
    """Get heuristic vs learned router comparison (Experiment 5)."""
    data = router_trainer.load_training_data()
    if data is None:
        raise HTTPException(status_code=404, detail="No training data available. Run full-grid first.")

    if not strategy_classifier._trained and not strategy_classifier.load_model():
        # Try training
        train_result = strategy_classifier.train(data)
        if train_result.get("status") != "trained":
            return {
                "status": "not_trained",
                "message": "Could not train classifier. Run full-grid and train-router first.",
            }

    eval_result = strategy_classifier.evaluate(data)

    # Add strategy distribution comparison
    dist = router_trainer.get_strategy_distribution()
    heuristic_stats = router_trainer.compute_heuristic_accuracy()

    return {
        "experiment": "router_comparison",
        "heuristic_accuracy": round(heuristic_stats["accuracy"], 4),
        "learned_accuracy": eval_result.get("learned_accuracy", 0),
        "cv_accuracy": eval_result.get("cv_accuracy", 0),
        "improvement": eval_result.get("improvement", 0),
        "confusion_matrix": eval_result.get("confusion_matrix", []),
        "classification_report": eval_result.get("classification_report", {}),
        "n_samples": eval_result.get("n_samples", 0),
        "oracle_strategy_distribution": dist,
        "interpretation": _interpret_router_results(eval_result),
    }


def _interpret_router_results(eval_result: Dict[str, Any]) -> str:
    """Generate human-readable interpretation of router comparison."""
    improvement = eval_result.get("improvement", 0)
    learned = eval_result.get("learned_accuracy", 0)
    heuristic = eval_result.get("heuristic_accuracy", 0)

    if improvement > 0.05:
        return (
            f"学习型路由器显著优于启发式规则（{learned:.1%} vs {heuristic:.1%}，"
            f"提升 {improvement:.1%}）。关键特征（如 question_type 和 entity_count）"
            f"提供了查询复杂度之外的额外信息。"
        )
    elif improvement > 0:
        return (
            f"学习型路由器略优于启发式规则（{learned:.1%} vs {heuristic:.1%}），"
            f"但提升幅度有限（+{improvement:.1%}）。启发式阈值可能已经捕获了大部分信息。"
        )
    elif improvement > -0.05:
        return (
            f"学习型路由器与启发式规则表现接近（{learned:.1%} vs {heuristic:.1%}），"
            f"说明查询复杂度是策略选择的主要决定因素。"
        )
    else:
        return (
            f"学习型路由器目前不如启发式规则（{learned:.1%} vs {heuristic:.1%}）。"
            f"可能需要更多训练数据或更丰富的特征。"
        )
