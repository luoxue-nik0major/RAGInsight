"""
Tests for learned strategy classifier.
"""
import pytest
import numpy as np
from app.services.strategy_classifier import (
    StrategyClassifier,
    FEATURE_NAMES,
    QUESTION_TYPES,
    STRATEGIES,
)

# Minimal training samples for testing
def make_samples(n: int = 30) -> list:
    samples = []
    strategies = ["vector", "hybrid", "graph"]
    question_types = ["factual", "definition", "comparative", "multi_hop"]
    for i in range(n):
        qt = question_types[i % len(question_types)]
        score = 0.2 + (i % 3) * 0.25  # spread across complexity
        is_zh = 1 if i >= n // 2 else 0
        oracle = "vector" if score < 0.3 else ("hybrid" if score < 0.7 else "graph")
        samples.append({
            "query_id": 1000 + i,
            "query": f"Test query {i}",
            "language": "zh" if is_zh else "en",
            "complexity_score": score,
            "question_type": qt,
            "features": {
                "length_score": 0.3 + i * 0.02,
                "sentence_score": 0.2 + i * 0.01,
                "entity_score": 0.1 + i * 0.03,
                "relation_score": 0.15 + i * 0.02,
                "semantic_score": 0.25 + i * 0.01,
                "hop_demand_score": 0.1 + i * 0.02,
                "length": 50 + i * 5,
                "sentence_count": 1 + i // 10,
                "entity_count": i % 5,
                "relation_count": i % 3,
            },
            "oracle_strategy": oracle,
            "oracle_quality": 0.5 + 0.3 * np.random.random(),
        })
    return samples


class TestFeatureExtraction:
    def test_extract_features_shape(self):
        clf = StrategyClassifier()
        samples = make_samples(10)
        X = clf._extract_features(samples)
        # 7 continuous + 6 question_type one-hot + 1 language = 14 features
        assert X.shape == (10, 14)
        assert X.dtype == np.float64

    def test_extract_features_language_flag(self):
        clf = StrategyClassifier()
        samples = [{
            "query_id": 1,
            "query": "test",
            "language": "zh",
            "complexity_score": 0.5,
            "question_type": "factual",
            "features": {k: 0.5 for k in FEATURE_NAMES},
            "oracle_strategy": "vector",
        }]
        X = clf._extract_features(samples)
        # Last feature should be 1.0 for Chinese
        assert X[0, -1] == 1.0

    def test_extract_features_english(self):
        clf = StrategyClassifier()
        samples = [{
            "query_id": 1,
            "query": "test",
            "language": "en",
            "complexity_score": 0.5,
            "question_type": "factual",
            "features": {k: 0.5 for k in FEATURE_NAMES},
            "oracle_strategy": "vector",
        }]
        X = clf._extract_features(samples)
        assert X[0, -1] == 0.0

    def test_get_feature_labels(self):
        clf = StrategyClassifier()
        labels = clf._get_feature_labels()
        assert len(labels) == 14
        assert labels[0] == "complexity_score"
        assert "qt_factual" in labels
        assert "qt_multi_hop" in labels
        assert "is_chinese" in labels


class TestTraining:
    def test_train_with_samples(self):
        clf = StrategyClassifier()
        samples = make_samples(30)
        result = clf.train(samples)
        assert result["status"] == "trained"
        assert result["n_samples"] == 30
        assert 0.0 <= result["cv_accuracy"] <= 1.0
        assert len(result["feature_importance"]) > 0

    def test_train_insufficient_data(self):
        clf = StrategyClassifier()
        result = clf.train(make_samples(3))
        assert result["status"] == "insufficient_data"

    def test_train_then_predict(self):
        clf = StrategyClassifier()
        samples = make_samples(30)
        clf.train(samples)

        pred = clf.predict({
            "complexity_score": 0.85,
            "question_type": "multi_hop",
            "features": {
                "length_score": 0.7,
                "sentence_score": 0.5,
                "entity_score": 0.8,
                "relation_score": 0.7,
                "semantic_score": 0.9,
                "hop_demand_score": 0.7,
            },
        }, "What is the relationship between A, B, and C?")
        assert "recommended_strategy" in pred
        assert pred["recommended_strategy"] in STRATEGIES
        assert "confidence_scores" in pred
        assert pred["router_mode"] == "learned"

    def test_predict_without_training_falls_back(self):
        clf = StrategyClassifier()
        pred = clf.predict({
            "complexity_score": 0.15,
            "question_type": "factual",
            "features": {},
        })
        assert pred["recommended_strategy"] == "vector"
        assert pred["router_mode"] == "heuristic_fallback"


class TestEvaluation:
    def test_evaluate_on_training_data(self):
        clf = StrategyClassifier()
        samples = make_samples(30)
        clf.train(samples)

        eval_result = clf.evaluate(samples)
        assert "learned_accuracy" in eval_result
        assert "heuristic_accuracy" in eval_result
        assert "cv_accuracy" in eval_result
        assert "confusion_matrix" in eval_result
        assert 0.0 <= eval_result["learned_accuracy"] <= 1.0
        # Note: improvement may not be positive on small synthetic data

    def test_evaluate_without_training(self):
        clf = StrategyClassifier()
        result = clf.evaluate(make_samples(10))
        assert result["status"] == "not_trained"


class TestModelPersistence:
    def test_save_and_load(self, tmp_path):
        import os
        import pickle
        import importlib
        from app.services import strategy_classifier as sc_module

        # Temporarily override MODEL_DIR
        original_dir = sc_module.MODEL_DIR
        sc_module.MODEL_DIR = str(tmp_path)

        try:
            clf = StrategyClassifier()
            samples = make_samples(30)
            clf.train(samples)

            # Create a new classifier and load
            clf2 = StrategyClassifier()
            assert clf2.load_model() is True
            assert clf2._trained is True

            # Verify it can predict
            pred = clf2.predict({
                "complexity_score": 0.5,
                "question_type": "comparative",
                "features": {k: 0.5 for k in FEATURE_NAMES},
            }, "Compare A and B")
            assert pred["recommended_strategy"] in STRATEGIES
        finally:
            sc_module.MODEL_DIR = original_dir
            # Clean up
            pkl_path = os.path.join(str(tmp_path), "strategy_classifier.pkl")
            if os.path.exists(pkl_path):
                os.remove(pkl_path)

    def test_load_nonexistent_model(self):
        clf = StrategyClassifier()
        import importlib
        from app.services import strategy_classifier as sc_module

        original_dir = sc_module.MODEL_DIR
        sc_module.MODEL_DIR = "/nonexistent/path/12345"
        try:
            assert clf.load_model() is False
        finally:
            sc_module.MODEL_DIR = original_dir
