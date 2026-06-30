"""Tests for root cause analyzer."""
import pytest
from app.services.root_cause import root_cause_analyzer


class TestRootCauseAnalysis:
    def test_empty_steps_and_alerts(self):
        """No steps/alerts should return None."""
        result = root_cause_analyzer.analyze([], [])
        assert result is None

    def test_empty_results(self):
        """Empty retrieval results → knowledge_incomplete."""
        steps = [
            {"step_type": "vector_retrieve", "id": 1, "quality_score": 0.0},
        ]
        alerts = [
            {"alert_type": "empty_results", "severity": "error", "step_id": 1},
        ]
        result = root_cause_analyzer.analyze(steps, alerts)
        assert result is not None
        assert result["root_cause_type"] == "incomplete_knowledge"
        assert 1 in result["involved_step_ids"]

    def test_low_relevance_and_coverage(self):
        """Low relevance + low coverage → retrieval_failure."""
        steps = [
            {"step_type": "vector_retrieve", "id": 1, "quality_score": 0.2},
        ]
        alerts = [
            {"alert_type": "low_relevance", "severity": "warning", "step_id": 1},
            {"alert_type": "low_coverage", "severity": "warning", "step_id": 1},
        ]
        result = root_cause_analyzer.analyze(steps, alerts)
        assert result is not None
        assert result["root_cause_type"] == "retrieval_failure"

    def test_hallucination_risk(self):
        """Invalid citations → hallucination_risk."""
        steps = [
            {"step_type": "answer_generate", "id": 2, "quality_score": 0.3},
        ]
        alerts = [
            {"alert_type": "hallucination_risk", "severity": "warning", "step_id": 2},
        ]
        result = root_cause_analyzer.analyze(steps, alerts)
        assert result is not None
        assert result["root_cause_type"] == "hallucination_risk"

    def test_context_too_long(self):
        """Context too long → strategy_mismatch."""
        steps = [
            {"step_type": "context_build", "id": 1, "quality_score": 0.0},
        ]
        alerts = [
            {"alert_type": "context_too_long", "severity": "error", "step_id": 1},
        ]
        result = root_cause_analyzer.analyze(steps, alerts)
        assert result is not None
        assert result["root_cause_type"] == "strategy_mismatch"

    def test_explanation_and_suggestions(self):
        steps = [
            {"step_type": "vector_retrieve", "id": 1, "quality_score": 0.0},
        ]
        alerts = [
            {"alert_type": "empty_results", "severity": "error", "step_id": 1},
        ]
        result = root_cause_analyzer.analyze(steps, alerts)
        assert len(result["explanation"]) > 0
        assert len(result["suggestions"]) > 0
        assert result["alert_count"] == 1
        assert result["error_count"] == 1
        assert result["warning_count"] == 0
