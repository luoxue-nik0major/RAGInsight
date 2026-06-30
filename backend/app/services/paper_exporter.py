"""
Paper figure and table exporter for RAGInsight.

Generates publication-quality outputs:
  - LaTeX-formatted tables (.tex)
  - Matplotlib figures (.png, 300dpi)
  - Statistical annotations (bootstrap CI, significance markers)
"""
import os
import json
import numpy as np
from typing import Dict, Any, List, Optional

PAPER_FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments",
    "paper_figures",
)


class PaperExporter:
    """Export experiment results as paper-ready tables and figures."""

    def __init__(self):
        self.output_dir = PAPER_FIGURES_DIR

    def ensure_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    # ── LaTeX Table Generation ──────────────────────────────────────────

    def generate_fault_detection_table(
        self,
        fault_metrics: Dict[str, Any],
        filename: str = "table_fault_detection.tex",
    ) -> str:
        """Generate LaTeX table for Experiment 1: Fault Detection."""
        self.ensure_output_dir()

        per_type = fault_metrics.get("per_alert_type", {})
        macro = fault_metrics.get("macro", {})
        micro = fault_metrics.get("micro", {})

        rows = []
        for alert_type, m in per_type.items():
            rows.append(
                f"    {alert_type.replace('_', '\\_')} & "
                f"{m['precision']:.3f} & {m['recall']:.3f} & {m['f1']:.3f} & "
                f"{m['tp']} & {m['fp']} & {m['fn']} \\\\"
            )

        latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{RAG Failure Detection Accuracy by Alert Type}}
\\label{{tab:fault_detection}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Alert Type}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1}} \\\\
\\midrule
{chr(10).join(rows)}
\\midrule
\\textbf{{Macro Avg}} & {macro.get('precision', 0):.3f} & {macro.get('recall', 0):.3f} & {macro.get('f1', 0):.3f} \\\\
\\textbf{{Micro Avg}} & {micro.get('precision', 0):.3f} & {micro.get('recall', 0):.3f} & {micro.get('f1', 0):.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(latex)
        return latex

    def generate_strategy_comparison_table(
        self,
        strategy_metrics: Dict[str, Any],
        filename: str = "table_strategy_comparison.tex",
    ) -> str:
        """Generate LaTeX table for Experiment 2: Cross-Strategy Comparison."""
        self.ensure_output_dir()

        strategies = strategy_metrics.get("strategies", {})

        header = " & ".join(strategies.keys())
        latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Cross-Strategy Retrieval and Answer Quality Comparison}}
\\label{{tab:strategy_comparison}}
\\begin{{tabular}}{{l{''.join('c' for _ in strategies)}}}
\\toprule
\\textbf{{Metric}} & {header} \\\\
\\midrule
"""
        metrics_map = [
            ("\\# Runs", "count"),
            ("Avg Relevance", "avg_relevance"),
            ("Avg Coverage", "avg_coverage"),
            ("Avg Diversity", "avg_diversity"),
            ("Avg Answer Quality", "avg_answer_quality"),
            ("Avg Faithfulness", "avg_faithfulness"),
            ("Avg Answer Relevance", "avg_answer_relevance"),
            ("Avg Duration (ms)", "avg_duration_ms"),
        ]

        for label, key in metrics_map:
            values = []
            for strategy in strategies:
                v = strategies[strategy].get(key, 0)
                if key == "count":
                    values.append(str(int(v)))
                elif key == "avg_duration_ms":
                    values.append(f"{v:.0f}")
                else:
                    values.append(f"{v:.3f}")
            latex += f"    {label} & {' & '.join(values)} \\\\\n"

        latex += """\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(latex)
        return latex

    def generate_router_comparison_table(
        self,
        router_metrics: Dict[str, Any],
        filename: str = "table_router_comparison.tex",
    ) -> str:
        """Generate LaTeX table for Experiment 5: Router Comparison."""
        self.ensure_output_dir()

        latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Query-to-Strategy Router Comparison}}
\\label{{tab:router_comparison}}
\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Method}} & \\textbf{{Accuracy}} \\\\
\\midrule
Heuristic (Threshold) & {router_metrics.get('heuristic_accuracy', 0):.3f} \\\\
Learned (Logistic Regression) & {router_metrics.get('learned_accuracy', 0):.3f} \\\\
Cross-Validation (5-fold) & {router_metrics.get('cv_accuracy', 0):.3f} \\\\
Oracle (Upper Bound) & 1.000 \\\\
Random Baseline & 0.333 \\\\
Always-Vector Baseline & — \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(latex)
        return latex

    # ── Matplotlib Figure Generation ────────────────────────────────────

    def generate_strategy_boxplot(
        self,
        results: List[Dict[str, Any]],
        metric: str = "relevance",
        filename: str = "fig_strategy_boxplot.png",
    ) -> Optional[str]:
        """Generate box plot comparing strategies on a metric."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self.ensure_output_dir()

            strategy_values: Dict[str, List[float]] = {}
            for r in results:
                strategy = r.get("strategy", "unknown")
                if strategy not in strategy_values:
                    strategy_values[strategy] = []

                for step in r.get("steps", []):
                    if step.get("step_type") == "vector_retrieve":
                        qm = step.get("output_data", {}).get("quality_metrics", {})
                        val = qm.get(metric, 0)
                        strategy_values[strategy].append(val)
                        break
                else:
                    strategy_values[strategy].append(0)

            fig, ax = plt.subplots(figsize=(6, 4))
            data = [strategy_values.get(s, []) for s in ["vector", "hybrid", "graph"]]
            labels = ["Vector", "Hybrid", "Graph"]

            bp = ax.boxplot(
                [d for d in data if d],
                labels=[l for d, l in zip(data, labels) if d],
                patch_artist=True,
            )
            for patch, color in zip(bp["boxes"], ["#6366f1", "#f59e0b", "#10b981"]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"Strategy Comparison: {metric.replace('_', ' ').title()}")
            ax.grid(axis="y", alpha=0.3)

            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return filepath
        except Exception as e:
            print(f"Failed to generate boxplot: {e}")
            return None

    def generate_feature_importance_bar(
        self,
        feature_importance: Dict[str, float],
        filename: str = "fig_feature_importance.png",
    ) -> Optional[str]:
        """Generate horizontal bar chart of feature importance for router."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self.ensure_output_dir()

            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]

            fig, ax = plt.subplots(figsize=(8, 5))
            names = [f[0] for f in sorted_features]
            values = [f[1] for f in sorted_features]

            colors = plt.cm.Blues([0.3 + 0.7 * v / max(values, 0.01) for v in values])
            ax.barh(names, values, color=colors)
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance: Strategy Router")
            ax.invert_yaxis()

            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return filepath
        except Exception as e:
            print(f"Failed to generate feature importance chart: {e}")
            return None

    def generate_complexity_radar(
        self,
        complexity_data: List[Dict[str, Any]],
        filename: str = "fig_complexity_radar.png",
    ) -> Optional[str]:
        """Generate radar chart comparing complexity profiles by strategy."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self.ensure_output_dir()

            dimensions = [
                "length_score", "sentence_score", "entity_score",
                "relation_score", "semantic_score", "hop_demand_score",
            ]

            # Aggregate by oracle strategy
            strategy_profiles: Dict[str, List[float]] = {
                s: [] for s in ["vector", "hybrid", "graph"]
            }
            for sample in complexity_data:
                oracle = sample.get("oracle_strategy", "vector")
                features = sample.get("features", {})
                avg = np.mean([features.get(d, 0) for d in dimensions])
                strategy_profiles[oracle].append(avg)

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

            angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
            angles += angles[:1]

            for strategy, color in [("vector", "#6366f1"), ("hybrid", "#f59e0b"), ("graph", "#10b981")]:
                vals = strategy_profiles.get(strategy, [])
                if vals:
                    avg_vals = []
                    for d in dimensions:
                        vals_d = [
                            sample.get("features", {}).get(d, 0)
                            for sample in complexity_data
                            if sample.get("oracle_strategy") == strategy
                        ]
                        avg_vals.append(np.mean(vals_d) if vals_d else 0)
                    avg_vals += avg_vals[:1]
                    ax.fill(angles, avg_vals, alpha=0.1, color=color)
                    ax.plot(angles, avg_vals, color=color, linewidth=2, label=strategy)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([d.replace("_score", "").title() for d in dimensions])
            ax.set_title("Complexity Profiles by Oracle Strategy")
            ax.legend(loc="upper right")

            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return filepath
        except Exception as e:
            print(f"Failed to generate radar chart: {e}")
            return None

    def export_all(
        self,
        experiment_metrics: Dict[str, Any],
        router_metrics: Optional[Dict[str, Any]] = None,
        results: Optional[List[Dict[str, Any]]] = None,
        training_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Export all paper figures and tables."""
        self.ensure_output_dir()
        outputs = {}

        # Tables
        exp1 = experiment_metrics.get("experiment_1_fault_detection")
        if exp1:
            outputs["fault_detection_table"] = self.generate_fault_detection_table(exp1)

        exp2 = experiment_metrics.get("experiment_2_strategy_comparison")
        if exp2:
            outputs["strategy_comparison_table"] = self.generate_strategy_comparison_table(exp2)

        if router_metrics:
            outputs["router_comparison_table"] = self.generate_router_comparison_table(router_metrics)

        # Figures
        if results:
            outputs["strategy_boxplot"] = self.generate_strategy_boxplot(results)

        if training_data:
            outputs["complexity_radar"] = self.generate_complexity_radar(training_data)

        # Feature importance from trained classifier
        try:
            from app.services.strategy_classifier import strategy_classifier
            if strategy_classifier._trained and strategy_classifier._feature_importance:
                outputs["feature_importance"] = self.generate_feature_importance_bar(
                    strategy_classifier._feature_importance
                )
        except Exception:
            pass

        outputs["output_dir"] = self.output_dir
        outputs["files_generated"] = [
            str(v) for k, v in outputs.items()
            if isinstance(v, str) and k != "output_dir"
        ]

        return outputs


paper_exporter = PaperExporter()
