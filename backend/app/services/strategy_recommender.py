"""
Strategy recommendation engine.
Maps complexity score to retrieval strategy using rule-based matrix.
Supports both heuristic (rule-based) and learned (ML classifier) modes.
"""
from typing import Dict, Any, List

from app.core.config import get_settings


class StrategyRecommender:
    """Recommend retrieval strategy based on query complexity (heuristic)."""

    STRATEGY_THRESHOLDS = [
        (0.30, "vector", "向量检索"),
        (0.70, "hybrid", "混合检索"),
        (1.00, "graph", "图检索"),
    ]

    STRATEGY_DESCRIPTIONS = {
        "vector": "基于语义相似度的向量检索，适合概念性、语义丰富的查询。",
        "hybrid": "向量检索 + 关键词检索融合，适合需要精确匹配和语义理解的查询。",
        "graph": "基于知识图谱的多跳推理检索，适合复杂关系查询。",
    }

    REASON_TEMPLATES = {
        "vector": (
            "查询复杂度较低（{score:.0%}），属于 {question_type} 类型，语义较为直接。"
            "向量检索能够高效匹配语义相似的内容。"
        ),
        "hybrid": (
            "查询复杂度中等（{score:.0%}），包含 {entity_count} 个实体和 {relation_count} 个关系词，"
            "属于 {question_type} 类型。混合检索可兼顾语义理解与精确匹配。"
        ),
        "graph": (
            "查询复杂度较高（{score:.0%}），涉及多实体关系或深层推理，"
            "属于 {question_type} 类型。图检索支持多跳推理，适合此类查询。"
        ),
    }

    def recommend(self, complexity_score: float, query: str = "", features: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recommend strategy and generate reason."""
        strategy_id = "vector"
        strategy_name = "向量检索"
        for threshold, sid, sname in self.STRATEGY_THRESHOLDS:
            if complexity_score < threshold:
                strategy_id = sid
                strategy_name = sname
                break

        template = self.REASON_TEMPLATES[strategy_id]
        reason = template.format(
            score=complexity_score,
            question_type=features.get("question_type", "unknown") if features else "unknown",
            entity_count=features.get("entity_count", 0) if features else 0,
            relation_count=features.get("relation_count", 0) if features else 0,
        )

        alternatives: List[Dict[str, str]] = []
        for threshold, sid, sname in self.STRATEGY_THRESHOLDS:
            if sid != strategy_id:
                alternatives.append({
                    "id": sid,
                    "name": sname,
                    "description": self.STRATEGY_DESCRIPTIONS[sid],
                })

        return {
            "recommended_strategy": strategy_id,
            "recommended_strategy_name": strategy_name,
            "complexity_score": complexity_score,
            "reason": reason,
            "alternatives": alternatives,
            "router_mode": "heuristic",
        }


class LearnedStrategyRecommender:
    """Recommend retrieval strategy using trained ML classifier."""

    def recommend(self, complexity_result: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        """Use the learned classifier to predict optimal strategy."""
        from app.services.strategy_classifier import strategy_classifier

        pred = strategy_classifier.predict(complexity_result, query)

        strategy_id = pred["recommended_strategy"]
        confidence = pred.get("confidence_scores", {}).get(strategy_id, 0.0)

        # Build reason from classifier output
        top_features = sorted(
            pred.get("feature_importance", {}).items(),
            key=lambda x: x[1], reverse=True
        )[:3]
        feature_reason = ", ".join(f"{k}={v:.3f}" for k, v in top_features)

        strategy_names = {
            "vector": "向量检索",
            "hybrid": "混合检索",
            "graph": "图检索",
        }
        strategy_name = strategy_names.get(strategy_id, strategy_id)

        alternatives: List[Dict[str, str]] = []
        for sid, sname in strategy_names.items():
            if sid != strategy_id:
                alternatives.append({
                    "id": sid,
                    "name": sname,
                    "description": StrategyRecommender.STRATEGY_DESCRIPTIONS[sid],
                })

        reason = (
            f"学习型路由器预测（置信度 {confidence:.0%}）。"
            f"关键特征: {feature_reason}。"
        )

        return {
            "recommended_strategy": strategy_id,
            "recommended_strategy_name": strategy_name,
            "complexity_score": complexity_result.get("complexity_score", 0),
            "reason": reason,
            "alternatives": alternatives,
            "router_mode": "learned",
            "confidence_scores": pred.get("confidence_scores", {}),
            "feature_importance": pred.get("feature_importance", {}),
        }


def get_recommender() -> StrategyRecommender:
    """Factory that returns the appropriate recommender based on config."""
    settings = get_settings()
    if settings.router_mode == "learned":
        # Check if model is available; fallback to heuristic if not
        from app.services.strategy_classifier import strategy_classifier
        if strategy_classifier._trained or strategy_classifier.load_model():
            # Return a wrapper that delegates to LearnedStrategyRecommender
            return _RouterWrapper()
    return strategy_recommender


class _RouterWrapper:
    """Wrapper that uses learned mode when available, falls back to heuristic."""

    def recommend(self, complexity_score: float, query: str = "", features: Dict[str, Any] = None) -> Dict[str, Any]:
        from app.services.strategy_classifier import strategy_classifier

        # Build complexity_result from legacy args
        complexity_result = {
            "complexity_score": complexity_score,
            "question_type": features.get("question_type", "factual") if features else "factual",
            "features": features or {},
        }

        if strategy_classifier._trained or strategy_classifier.load_model():
            learned = LearnedStrategyRecommender()
            return learned.recommend(complexity_result, query)

        return strategy_recommender.recommend(complexity_score, query, features)


strategy_recommender = StrategyRecommender()
