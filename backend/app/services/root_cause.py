"""
Lightweight root cause analysis rule engine.
Analyzes session data to diagnose fault root causes.
"""
from typing import List, Dict, Any, Optional
from app.core.models import StepType, AlertSeverity


class RootCauseType(str):
    RETRIEVAL_FAILURE = "retrieval_failure"
    INCOMPLETE_KNOWLEDGE = "incomplete_knowledge"
    HALLUCINATION_RISK = "hallucination_risk"
    STRATEGY_MISMATCH = "strategy_mismatch"
    NO_ISSUE = "no_issue"


class RootCauseAnalyzer:
    """Analyze session data to identify root causes of faults."""

    @staticmethod
    def analyze(steps: List[Dict[str, Any]], alerts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Analyze steps and alerts to produce a root cause diagnosis.
        Returns None if no significant issues found.
        """
        if not alerts:
            return None

        error_alerts = [a for a in alerts if a.get("severity") == "error"]
        warning_alerts = [a for a in alerts if a.get("severity") == "warning"]

        # Collect alert types
        alert_types = set(a.get("alert_type") for a in alerts)

        # Determine root cause type
        root_cause_type = RootCauseType.NO_ISSUE
        involved_steps = []
        explanation = ""
        suggestions = []

        if "empty_results" in alert_types:
            root_cause_type = RootCauseType.INCOMPLETE_KNOWLEDGE
            involved_steps = [s for s in steps if s.get("step_type") == "vector_retrieve"]
            explanation = "检索结果为空，说明知识库中缺少与查询相关的文档。"
            suggestions = [
                "扩展知识库，添加与查询主题相关的文档",
                "检查查询是否过于具体或使用了知识库中未出现的术语",
            ]

        elif "low_relevance" in alert_types and "low_coverage" in alert_types:
            root_cause_type = RootCauseType.RETRIEVAL_FAILURE
            involved_steps = [s for s in steps if s.get("step_type") == "vector_retrieve"]
            explanation = "检索结果相关性低且未覆盖查询关键实体，检索器未能召回相关文档。"
            suggestions = [
                "尝试使用混合检索策略（向量+关键词）",
                "增加检索数量（top_k）",
                "检查embedding模型是否与查询领域匹配",
            ]

        elif "low_relevance" in alert_types:
            root_cause_type = RootCauseType.STRATEGY_MISMATCH
            involved_steps = [s for s in steps if s.get("step_type") == "vector_retrieve"]
            explanation = "检索结果相关性较低，当前检索策略可能不适合该查询类型。"
            suggestions = [
                "尝试切换检索策略",
                "对查询进行改写或扩展",
            ]

        elif "context_too_long" in alert_types:
            root_cause_type = RootCauseType.STRATEGY_MISMATCH
            involved_steps = [s for s in steps if s.get("step_type") == "context_build"]
            explanation = "构建的上下文超过模型token限制，需要精简检索结果。"
            suggestions = [
                "减少检索片段数量",
                "使用重排序策略只保留最相关的片段",
                "对检索结果进行摘要压缩",
            ]

        elif "hallucination_risk" in alert_types:
            root_cause_type = RootCauseType.HALLUCINATION_RISK
            involved_steps = [s for s in steps if s.get("step_type") == "answer_generate"]
            explanation = "生成的答案包含无法追溯到检索内容的引用，存在幻觉风险。"
            suggestions = [
                "要求模型更严格地基于检索内容回答",
                "检查检索片段是否包含足够信息",
                "增加检索片段数量以提高覆盖面",
            ]

        elif warning_alerts:
            # Generic warning - pick the most severe
            root_cause_type = RootCauseType.RETRIEVAL_FAILURE
            involved_steps = [s for s in steps if s.get("step_type") in ("vector_retrieve", "context_build")]
            explanation = "检索过程中检测到潜在质量问题，可能影响答案准确性。"
            suggestions = [
                "检查检索结果质量指标",
                "考虑更换检索策略",
            ]

        if root_cause_type == RootCauseType.NO_ISSUE:
            return None

        return {
            "root_cause_type": root_cause_type,
            "root_cause_label": RootCauseAnalyzer._get_label(root_cause_type),
            "severity": "error" if error_alerts else "warning",
            "involved_step_ids": [s.get("id") for s in involved_steps],
            "explanation": explanation,
            "suggestions": suggestions,
            "alert_count": len(alerts),
            "error_count": len(error_alerts),
            "warning_count": len(warning_alerts),
        }

    @staticmethod
    def _get_label(cause_type: str) -> str:
        labels = {
            RootCauseType.RETRIEVAL_FAILURE: "检索失败",
            RootCauseType.INCOMPLETE_KNOWLEDGE: "知识不完整",
            RootCauseType.HALLUCINATION_RISK: "幻觉风险",
            RootCauseType.STRATEGY_MISMATCH: "策略不匹配",
            RootCauseType.NO_ISSUE: "无问题",
        }
        return labels.get(cause_type, "未知")


root_cause_analyzer = RootCauseAnalyzer()
