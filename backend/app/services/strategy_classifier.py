"""
Learned query-to-strategy router using logistic regression.

Trains on oracle-labeled data from full-grid experiments.
Predicts optimal retrieval strategy from query complexity features.
"""
import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments",
)

FEATURE_NAMES = [
    "complexity_score",
    "length_score",
    "sentence_score",
    "entity_score",
    "relation_score",
    "semantic_score",
    "hop_demand_score",
]

QUESTION_TYPES = ["factual", "definition", "list", "comparative", "causal", "multi_hop"]
STRATEGIES = ["vector", "hybrid", "graph"]


class StrategyClassifier:
    """Logistic regression classifier for query-to-strategy routing."""

    def __init__(self):
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self._feature_importance: Dict[str, float] = {}
        self._cv_accuracy: float = 0.0
        self._cv_scores: List[float] = []
        self._trained: bool = False

    def _extract_features(self, samples: List[Dict[str, Any]]) -> np.ndarray:
        """Extract feature matrix from training samples."""
        X = []
        for sample in samples:
            feats = sample.get("features", sample)
            row = [sample.get("complexity_score", sample.get("complexity_score", 0))]
            for name in FEATURE_NAMES[1:]:
                row.append(feats.get(name, 0))
            # Question type one-hot
            qt = sample.get("question_type", "factual")
            for qt_name in QUESTION_TYPES:
                row.append(1.0 if qt == qt_name else 0.0)
            # Language flag: 1 for Chinese, 0 for English
            lang = sample.get("language", "en")
            row.append(1.0 if lang == "zh" else 0.0)
            X.append(row)
        return np.array(X, dtype=np.float64)

    def _get_feature_labels(self) -> List[str]:
        labels = list(FEATURE_NAMES)
        for qt in QUESTION_TYPES:
            labels.append(f"qt_{qt}")
        labels.append("is_chinese")
        return labels

    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the logistic regression classifier.

        Args:
            training_data: List of samples from RouterTrainer, each with
                           'oracle_strategy' and feature fields.

        Returns:
            Dict with training results including CV accuracy and feature importance.
        """
        if len(training_data) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 10 samples, got {len(training_data)}",
            }

        X = self._extract_features(training_data)
        y = np.array([s["oracle_strategy"] for s in training_data])

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train logistic regression with cross-validation
        self.model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
            random_state=42,
        )

        # 5-fold stratified cross-validation
        try:
            n_folds = min(5, min(np.bincount(y_encoded)))
            if n_folds < 2:
                n_folds = 2

            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                self.model, X_scaled, y_encoded, cv=skf, scoring="accuracy"
            )
            self._cv_scores = [float(s) for s in cv_scores]
            self._cv_accuracy = float(np.mean(cv_scores))
        except Exception:
            self._cv_accuracy = 0.0
            self._cv_scores = []

        # Fit on all data
        self.model.fit(X_scaled, y_encoded)

        # Extract feature importance (coefficient magnitudes)
        feature_labels = self._get_feature_labels()
        if hasattr(self.model, "coef_"):
            coef_abs = np.abs(self.model.coef_).mean(axis=0)
            total = coef_abs.sum()
            if total > 0:
                self._feature_importance = {
                    feature_labels[i]: round(float(coef_abs[i] / total), 4)
                    for i in range(min(len(feature_labels), len(coef_abs)))
                }
            else:
                self._feature_importance = {label: 0.0 for label in feature_labels}

        self._trained = True

        # Save model
        self._save_model()

        return {
            "status": "trained",
            "cv_accuracy": self._cv_accuracy,
            "cv_scores": self._cv_scores,
            "cv_folds": len(self._cv_scores),
            "feature_importance": self._feature_importance,
            "n_samples": len(training_data),
            "class_distribution": {
                k: int(v) for k, v in zip(
                    *np.unique(y, return_counts=True)
                )
            },
        }

    def predict(self, complexity_result: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        """
        Predict optimal strategy for a query.

        Args:
            complexity_result: Output from complexity_analyzer.analyze()
            query: Original query string (for language detection)

        Returns:
            Dict with predicted strategy, confidence scores, and feature importance.
        """
        if not self._trained:
            # Fallback to heuristic
            score = complexity_result.get("complexity_score", 0)
            if score < 0.30:
                pred = "vector"
            elif score < 0.70:
                pred = "hybrid"
            else:
                pred = "graph"
            return {
                "recommended_strategy": pred,
                "confidence_scores": {"vector": 0.34, "hybrid": 0.33, "graph": 0.33},
                "feature_importance": {},
                "router_mode": "heuristic_fallback",
            }

        # Extract features
        features = complexity_result.get("features", {})
        from app.utils.text_utils import is_chinese_text

        sample = {
            "complexity_score": complexity_result.get("complexity_score", 0),
            "features": features,
            "question_type": complexity_result.get("question_type", "factual"),
            "language": "zh" if is_chinese_text(query) else "en",
        }

        X = self._extract_features([sample])
        X_scaled = self.scaler.transform(X)

        # Predict probabilities
        probas = self.model.predict_proba(X_scaled)[0]
        class_idx = int(np.argmax(probas))
        predicted_label = self.label_encoder.inverse_transform([class_idx])[0]

        # Build confidence dict
        confidence = {}
        for i, strategy in enumerate(self.label_encoder.classes_):
            confidence[strategy] = round(float(probas[i]), 4)

        return {
            "recommended_strategy": str(predicted_label),
            "confidence_scores": confidence,
            "feature_importance": self._feature_importance,
            "router_mode": "learned",
        }

    def evaluate(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate classifier performance on the training data (in-sample)."""
        if not self._trained:
            return {"status": "not_trained"}

        X = self._extract_features(training_data)
        y_true = np.array([s["oracle_strategy"] for s in training_data])
        y_heuristic = np.array([s.get("heuristic_strategy", "vector") for s in training_data])

        X_scaled = self.scaler.transform(X)
        y_pred = self.label_encoder.inverse_transform(self.model.predict(X_scaled))

        # Learned accuracy
        learned_acc = float((y_pred == y_true).mean())

        # Heuristic accuracy
        heuristic_acc = float((y_heuristic == y_true).mean())

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=STRATEGIES)

        # Classification report
        report = classification_report(y_true, y_pred, labels=STRATEGIES, output_dict=True, zero_division=0)

        return {
            "learned_accuracy": round(learned_acc, 4),
            "heuristic_accuracy": round(heuristic_acc, 4),
            "cv_accuracy": round(self._cv_accuracy, 4),
            "improvement": round(learned_acc - heuristic_acc, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "n_samples": len(training_data),
        }

    def _save_model(self) -> None:
        """Save model weights, scaler, and label encoder to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        import pickle

        model_path = os.path.join(MODEL_DIR, "strategy_classifier.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "label_encoder": self.label_encoder,
                "feature_importance": self._feature_importance,
                "cv_accuracy": self._cv_accuracy,
                "cv_scores": self._cv_scores,
            }, f)

    def load_model(self) -> bool:
        """Load saved model from disk. Returns True if successful."""
        import pickle

        model_path = os.path.join(MODEL_DIR, "strategy_classifier.pkl")
        if not os.path.exists(model_path):
            return False

        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.label_encoder = data["label_encoder"]
            self._feature_importance = data.get("feature_importance", {})
            self._cv_accuracy = data.get("cv_accuracy", 0.0)
            self._cv_scores = data.get("cv_scores", [])
            self._trained = True
            return True
        except Exception:
            return False


strategy_classifier = StrategyClassifier()
