# RAGInsight Paper Experiments — Reproduction Guide

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/yourusername/raginsight
cd raginsight

# 2. Configure API key
cp .env.docker .env
# Edit .env: set RAGINSIGHT_DEEPSEEK_API_KEY=sk-your-key

# 3. Install dependencies
cd backend
pip install -r requirements.txt

# 4. Initialize database
python -m alembic upgrade head

# 5. Run all paper experiments (generates figures in experiments/paper_figures/)
python scripts/run_paper_experiments.py --all
```

## Reproducing Individual Experiments

### E1: Fault Detection Accuracy
```bash
# Requires experiment results from full-grid run
python scripts/run_paper_experiments.py --experiments
# Then access: GET /api/experiments/metrics
```

### E2: Cross-Strategy Comparison
```bash
# Included in full-grid run above
# Metrics include: relevance, coverage, diversity, faithfulness, answer relevance, duration
```

### E3: Cross-Language Comparison
```bash
# Run experiments on both datasets:
# - squad_test_queries.json (English, 225 queries)
# - chinese_test_queries.json (Chinese, 50 queries)
# Metrics available at: GET /api/experiments/metrics
```

### E4: Causal Attribution Validation
```bash
# Run a single query through the pipeline, then:
# POST /api/sessions/{id}/causal-attribution
# Poll: GET /api/tasks/{task_id}/status
# Results: GET /api/sessions/{id}/causal-attribution
```

### E5: Learned Router Evaluation
```bash
# After full-grid experiments:
python scripts/run_paper_experiments.py --train
# Comparison: GET /api/experiments/router-comparison
```

## Paper Figures

All generated figures are in `backend/experiments/paper_figures/`:
- `table_fault_detection.tex` — Fault detection accuracy (Table 1)
- `table_strategy_comparison.tex` — Strategy comparison (Table 2)
- `table_router_comparison.tex` — Router comparison (Table 5)
- `fig_strategy_boxplot.png` — Strategy box plots (Figure 2)
- `fig_feature_importance.png` — Router feature importance (Figure 5)
- `fig_complexity_radar.png` — Complexity radar chart (Figure 3)

## Running Tests

```bash
cd backend
pytest tests/ -q  # 76+ tests
```

## Key Configuration

| Env Variable | Default | Description |
|---|---|---|
| `RAGINSIGHT_DEEPSEEK_API_KEY` | (required) | DeepSeek API key |
| `RAGINSIGHT_DEEPSEEK_MODEL` | deepseek-v4-flash | LLM model |
| `RAGINSIGHT_ROUTER_MODE` | heuristic | "heuristic" or "learned" |
| `RAGINSIGHT_EMBEDDING_MODEL` | BAAI/bge-small-zh-v1.5 | Embedding model |

## System Requirements

- Python 3.10+
- 8GB RAM
- DeepSeek API access (for answer generation and exact attribution)
- No GPU required (all embeddings run on CPU)
