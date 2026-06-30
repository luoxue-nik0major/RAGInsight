#!/usr/bin/env python
"""
One-click paper experiment reproduction script.

Usage:
    python scripts/run_paper_experiments.py --all          # Run everything
    python scripts/run_paper_experiments.py --experiments   # Run experiments only
    python scripts/run_paper_experiments.py --train         # Train router only
    python scripts/run_paper_experiments.py --export        # Export paper figures only

Output:
    experiments/paper_figures/  — LaTeX tables + PNG figures
    experiments/                — Experiment results + training data
"""
import os
import sys
import json
import asyncio
import argparse

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_queries(dataset: str = "squad"):
    """Load test queries from data files."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "data")
    filepath = os.path.join(data_dir, f"{dataset}_test_queries.json")
    if not os.path.exists(filepath):
        print(f"[WARN] Dataset not found: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", [])


async def run_experiments():
    """Run full-grid experiments on all datasets."""
    from app.services.experiment_runner import experiment_runner

    print("=" * 60)
    print("E1-E2: Running Full-Grid Experiments")
    print("=" * 60)

    all_results = []

    for dataset_name in ["squad", "chinese"]:
        queries = load_queries(dataset_name)
        if not queries:
            print(f"[SKIP] No queries for dataset: {dataset_name}")
            continue

        print(f"\n[{dataset_name}] Running {len(queries)} queries x 3 strategies = {len(queries) * 3} runs...")
        results = await experiment_runner.run_batch(queries, strategies=["vector", "hybrid", "graph"])
        all_results.extend(results)
        print(f"[{dataset_name}] Complete: {len(results)} results")

    print(f"\n[DONE] Total results: {len(all_results)}")
    return all_results


async def train_router():
    """Train strategy classifier on experiment results."""
    from app.services.router_trainer import router_trainer
    from app.services.strategy_classifier import strategy_classifier

    print("=" * 60)
    print("E5: Training Learned Strategy Router")
    print("=" * 60)

    # Load training data
    data = router_trainer.load_training_data()
    if data is None:
        # Generate from experiment results
        print("Generating training data from experiment results...")
        queries = load_queries("squad") + load_queries("chinese")
        if not queries:
            print("[ERROR] No queries available for training")
            return None
        data = await router_trainer.collect_training_data(queries, ["vector", "hybrid", "graph"])

    print(f"Training data: {len(data)} samples")

    # Heuristic baseline
    heuristic = router_trainer.compute_heuristic_accuracy()
    print(f"Heuristic baseline accuracy: {heuristic['accuracy']:.2%} ({heuristic['correct']}/{heuristic['total']})")

    # Strategy distribution
    dist = router_trainer.get_strategy_distribution()
    print(f"Oracle strategy distribution: {dist}")

    # Train classifier
    result = strategy_classifier.train(data)
    print(f"Training result: {result.get('status')}")
    if result.get("status") == "trained":
        print(f"CV Accuracy: {result['cv_accuracy']:.2%} (±{np.std(result['cv_scores']):.3f})")
        print(f"Top features:")
        for feat, imp in sorted(result["feature_importance"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feat}: {imp:.3f}")

    # Full evaluation
    eval_result = strategy_classifier.evaluate(data)
    print(f"\nLearned accuracy: {eval_result.get('learned_accuracy', 0):.2%}")
    print(f"Improvement over heuristic: {eval_result.get('improvement', 0):.3f}")
    print(f"Confusion matrix:\n{np.array(eval_result.get('confusion_matrix', []))}")

    return {"training_result": result, "evaluation": eval_result, "data": data}


def export_paper_figures():
    """Generate all paper figures and tables."""
    from app.services.paper_exporter import paper_exporter
    from app.routers.experiments import _compute_fault_detection_metrics, _compute_strategy_comparison

    print("=" * 60)
    print("Exporting Paper Figures and Tables")
    print("=" * 60)

    # Load latest experiment results
    experiments_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")
    files = sorted(
        [f for f in os.listdir(experiments_dir) if f.endswith(".jsonl")],
        reverse=True,
    )
    if files:
        with open(os.path.join(experiments_dir, files[0]), "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(results)} experiment results from {files[0]}")
    else:
        print("[WARN] No experiment results found")
        results = []

    # Try loading router training data
    training_data_path = os.path.join(experiments_dir, "router_training_data.json")
    training_data = None
    if os.path.exists(training_data_path):
        with open(training_data_path, "r", encoding="utf-8") as f:
            training_data = json.load(f)
        print(f"Loaded {len(training_data)} training samples")

    # Compute metrics
    metrics = {}
    if results:
        metrics["experiment_1_fault_detection"] = _compute_fault_detection_metrics(results)
        metrics["experiment_2_strategy_comparison"] = _compute_strategy_comparison(results)
        metrics["total_runs"] = len(results)

    # Try router comparison
    router_metrics = None
    try:
        from app.services.strategy_classifier import strategy_classifier
        from app.services.router_trainer import router_trainer
        if training_data:
            data = training_data
        else:
            data = router_trainer.load_training_data()

        if data and (strategy_classifier._trained or strategy_classifier.load_model()):
            eval_result = strategy_classifier.evaluate(data)
            heuristic = router_trainer.compute_heuristic_accuracy()
            router_metrics = {
                "heuristic_accuracy": heuristic["accuracy"],
                "learned_accuracy": eval_result.get("learned_accuracy", 0),
                "cv_accuracy": eval_result.get("cv_accuracy", 0),
                "improvement": eval_result.get("improvement", 0),
            }
    except Exception as e:
        print(f"[WARN] Router metrics unavailable: {e}")

    # Export
    outputs = paper_exporter.export_all(metrics, router_metrics, results, training_data)
    print(f"\n[DONE] Generated {len(outputs.get('files_generated', []))} files in {outputs['output_dir']}")
    for f in outputs.get("files_generated", []):
        print(f"  - {f}")

    return outputs


def main():
    parser = argparse.ArgumentParser(description="RAGInsight Paper Experiment Runner")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--experiments", action="store_true", help="Run full-grid experiments")
    parser.add_argument("--train", action="store_true", help="Train strategy router")
    parser.add_argument("--export", action="store_true", help="Export paper figures")
    args = parser.parse_args()

    if not any([args.all, args.experiments, args.train, args.export]):
        parser.print_help()
        return

    run_all = args.all

    if run_all or args.experiments:
        asyncio.run(run_experiments())

    if run_all or args.train:
        asyncio.run(train_router())

    if run_all or args.export:
        export_paper_figures()

    print("\n" + "=" * 60)
    print("All tasks complete!")
    print(f"Results: {os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments')}")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np
    main()
