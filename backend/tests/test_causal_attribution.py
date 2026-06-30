"""
Tests for causal attribution framework.
"""
import pytest
import numpy as np
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.causal_attribution import (
    CausalAttributionAnalyzer,
    ComponentType,
    InterventionType,
    InterventionResult,
    AttributionReport,
)


def make_mock_chunks(n: int = 3) -> list:
    return [
        {
            "content": f"This is chunk {i} with some test content about artificial intelligence.",
            "source": f"doc_{i}",
            "relevance_score": 0.8 - i * 0.1,
            "chunk_index": i,
            "id": 100 + i,
        }
        for i in range(n)
    ]


class TestCausalGraph:
    """Test causal DAG construction."""

    def test_build_basic_graph(self):
        analyzer = CausalAttributionAnalyzer()
        trace = {}
        graph = analyzer.build_causal_graph(trace)

        assert "nodes" in graph
        assert "edges" in graph
        assert "observables" in graph
        assert len(graph["nodes"]) == 6
        assert len(graph["edges"]) >= 6

        node_ids = {n["id"] for n in graph["nodes"]}
        assert "query" in node_ids
        assert "strategy" in node_ids
        assert "retrieval" in node_ids
        assert "context" in node_ids
        assert "llm" in node_ids
        assert "quality" in node_ids

    def test_graph_with_complexity(self):
        analyzer = CausalAttributionAnalyzer()
        trace = {"complexity": {"complexity_score": 0.75}}
        graph = analyzer.build_causal_graph(trace)

        assert graph["observables"]["query_complexity"] == 0.75

    def test_graph_with_answer_evaluation(self):
        analyzer = CausalAttributionAnalyzer()
        trace = {"answer_evaluation": {"combined_score": 0.85}}
        graph = analyzer.build_causal_graph(trace)

        assert graph["observables"]["answer_quality"] == 0.85

    def test_graph_with_full_trace(self):
        analyzer = CausalAttributionAnalyzer()
        trace = {
            "complexity": {"complexity_score": 0.5},
            "recommendation": {"recommended_strategy": "hybrid"},
            "quality_metrics": {"relevance": 0.8, "coverage": 0.6},
            "answer_evaluation": {"combined_score": 0.72},
        }
        graph = analyzer.build_causal_graph(trace)

        assert graph["observables"]["query_complexity"] == 0.5
        assert graph["observables"]["recommended_strategy"] == "hybrid"
        assert graph["observables"]["answer_quality"] == 0.72


class TestApproximateInterventions:
    """Test embedding-based approximate quality deltas."""

    def test_approx_chunks_single_removal(self):
        analyzer = CausalAttributionAnalyzer()
        # Mock embedding to return constant vectors
        with patch.object(analyzer, "_embed", return_value=np.ones((3, 512), dtype=np.float32)):
            chunks = make_mock_chunks(3)
            delta = analyzer._approx_intervene_chunks(
                "test answer", chunks, {0}, 0.8
            )
            # With constant embeddings, similarity ratio ≈ 1.0, delta ≈ 0
            assert abs(delta) < 0.1

    def test_approx_chunks_remove_all(self):
        analyzer = CausalAttributionAnalyzer()
        chunks = make_mock_chunks(2)
        delta = analyzer._approx_intervene_chunks(
            "test answer", chunks, {0, 1}, 0.8
        )
        assert delta <= 0  # Removing all chunks should degrade quality

    def test_approx_strategy_empty_new_chunks(self):
        analyzer = CausalAttributionAnalyzer()
        chunks = make_mock_chunks(3)
        delta = analyzer._approx_intervene_strategy(
            "test query", chunks, [], 0.8
        )
        assert delta == -0.8  # No chunks → quality drops to 0

    def test_approx_strategy_normal(self):
        analyzer = CausalAttributionAnalyzer()
        with patch.object(analyzer, "_embed", return_value=np.ones((3, 512), dtype=np.float32)):
            chunks = make_mock_chunks(3)
            new_chunks = make_mock_chunks(3)
            delta = analyzer._approx_intervene_strategy(
                "test query", chunks, new_chunks, 0.8
            )
            # With identical constant embeddings, avg_sim ≈ 1.0, delta ≈ 0
            assert abs(delta) < 0.1


class TestAttributionReport:
    """Test AttributionReport dataclass."""

    def test_empty_report(self):
        report = AttributionReport(
            session_id=1,
            query="test",
            original_strategy="vector",
            original_quality=0.5,
        )
        assert report.session_id == 1
        assert report.total_interventions == 0
        assert report.component_attributions == {}

    def test_report_with_interventions(self):
        report = AttributionReport(
            session_id=2,
            query="What is AI?",
            original_strategy="vector",
            original_quality=0.75,
            interventions=[
                InterventionResult(
                    component=ComponentType.STRATEGY,
                    intervention=InterventionType.SWITCH_STRATEGY,
                    params={"from": "vector", "to": "hybrid"},
                    original_quality=0.75,
                    perturbed_quality=0.82,
                    quality_delta=0.07,
                    attribution_score=0.35,
                    description="Switch vector→hybrid",
                ),
                InterventionResult(
                    component=ComponentType.CHUNK_SELECTION,
                    intervention=InterventionType.REMOVE_CHUNKS,
                    params={"removed_indices": [0]},
                    original_quality=0.75,
                    perturbed_quality=0.60,
                    quality_delta=-0.15,
                    attribution_score=0.65,
                    description="Remove chunk 0",
                ),
            ],
            component_attributions={"strategy": 0.35, "chunk_selection": 0.65},
            total_interventions=2,
        )
        assert report.total_interventions == 2
        assert abs(report.component_attributions["chunk_selection"] - 0.65) < 0.01
        assert len(report.interventions) == 2


class TestCausalAttributionAnalyzer:
    """Test the main analyzer class."""

    def test_init_defaults(self):
        analyzer = CausalAttributionAnalyzer()
        assert analyzer.exact_top_k == 3
        assert analyzer._embedding_fn is None

    def test_init_custom(self):
        analyzer = CausalAttributionAnalyzer(max_concurrent_llm=4, exact_top_k=5)
        assert analyzer.exact_top_k == 5

    def test_compute_answer_quality_with_chunks(self):
        analyzer = CausalAttributionAnalyzer()
        chunks = make_mock_chunks(2)
        quality = analyzer._compute_answer_quality(
            "What is AI?",
            "AI is artificial intelligence.",
            chunks,
        )
        assert 0.0 <= quality <= 1.0

    def test_compute_answer_quality_empty(self):
        analyzer = CausalAttributionAnalyzer()
        quality = analyzer._compute_answer_quality("query", "", [])
        assert quality == 0.0

    @pytest.mark.asyncio
    async def test_run_full_attribution_empty_chunks(self):
        analyzer = CausalAttributionAnalyzer()
        report = await analyzer.run_full_attribution(
            query="test",
            original_answer="test answer",
            chunks=[],
            original_strategy="vector",
            original_topk=5,
            session_trace={},
        )
        assert report.total_interventions == 0
        assert report.original_quality == 0.0

    @pytest.mark.asyncio
    async def test_run_full_attribution_basic(self):
        analyzer = CausalAttributionAnalyzer()
        chunks = make_mock_chunks(3)

        # Mock retriever to avoid ChromaDB access
        with patch("app.services.causal_attribution.retriever_registry") as mock_registry:
            mock_retriever = AsyncMock()
            mock_retriever.retrieve.return_value = {"chunks": make_mock_chunks(3), "total_found": 3}
            mock_registry.get.return_value = mock_retriever

            # Mock answer evaluator to avoid embedding model access
            with patch.object(analyzer, "_compute_answer_quality", return_value=0.75):
                # Mock embedding to avoid actual embedding calls
                with patch.object(analyzer, "_embed", return_value=np.ones((3, 512), dtype=np.float32)):
                    report = await analyzer.run_full_attribution(
                        query="What is AI?",
                        original_answer="AI is artificial intelligence technology.",
                        chunks=chunks,
                        original_strategy="vector",
                        original_topk=5,
                        session_trace={},
                    )

        assert report.original_quality == 0.75
        assert report.total_interventions > 0
        assert "causal_graph" in report.__dict__ or report.causal_graph is not None


class TestComponentType:
    def test_component_types(self):
        assert ComponentType.STRATEGY.value == "strategy"
        assert ComponentType.TOP_K.value == "top_k"
        assert ComponentType.CHUNK_SELECTION.value == "chunk_selection"
        assert ComponentType.CONTEXT_ASSEMBLY.value == "context_assembly"
        assert ComponentType.LLM_GENERATION.value == "llm_generation"

    def test_intervention_types(self):
        assert InterventionType.SWITCH_STRATEGY.value == "switch_strategy"
        assert InterventionType.CHANGE_TOPK.value == "change_topk"
        assert InterventionType.REMOVE_CHUNKS.value == "remove_chunks"
        assert InterventionType.COMPRESS_CONTEXT.value == "compress_context"
