"""
Query complexity analyzer.
Evaluates query complexity from multiple dimensions using local rule-based scoring.
Supports both English and Chinese queries.
No LLM calls — fully local computation.
"""
import re
from typing import Dict, Any, List

from app.utils.text_utils import is_chinese_text


class ComplexityAnalyzer:
    """Analyze query complexity across 5 dimensions."""

    RELATION_WORDS = {
        "compare", "comparison", "difference", "different", "between", "vs", "versus",
        "relationship", "relate", "connect", "connection", "link", "associate",
        "impact", "effect", "influence", "cause", "result", "lead", "due",
        "depend", "rely", "interact", "correlation", "relative",
        "similar", "unlike", "contrast", "against",
    }

    CHINESE_RELATION_WORDS = {
        "比较", "对比", "差异", "区别", "不同", "vs", "versus",
        "关系", "联系", "关联", "相关",
        "影响", "作用", "导致", "引起", "原因", "结果", "造成",
        "依赖", "依靠", "相互作用", "相关性", "相对",
        "相似", "相反", "对照",
    }

    QUESTION_TYPE_COMPLEXITY = {
        "factual": 0.15,
        "definition": 0.25,
        "list": 0.35,
        "comparative": 0.75,
        "causal": 0.80,
        "multi_hop": 0.95,
    }

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and return complexity score + feature breakdown.
        Returns dict with:
        - complexity_score (0~1)
        - question_type
        - features: {length, length_score, sentence_count, sentence_score,
                     entity_count, entity_score, relation_count, relation_score,
                     semantic_score, hop_demand_score}
        """
        if not query.strip():
            return {
                "complexity_score": 0.0,
                "question_type": "factual",
                "features": {},
            }

        q = query.strip()
        is_chinese = is_chinese_text(q)

        # 1. Length feature
        length = len(q)
        length_score = min(1.0, max(0.0, (length - 20) / 180.0))

        # 2. Sentence count (supports mixed punctuation)
        sentences = [s for s in re.split(r'[.!?。！？]+', q) if s.strip()]
        sentence_count = len(sentences)
        sentence_score = min(1.0, max(0.0, (sentence_count - 1) / 3.0))

        # 3. Entity count
        if is_chinese:
            entity_count, entities = self._extract_chinese_entities(q)
        else:
            entities = list(set(re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', q)))
            entity_count = len(entities)
        entity_score = min(1.0, max(0.0, (entity_count - 1) / 4.0))

        # 4. Relation words count
        if is_chinese:
            relation_count = self._count_chinese_relation_words(q)
        else:
            words = set(re.findall(r'\b[a-zA-Z]+\b', q.lower()))
            relation_count = len(words & self.RELATION_WORDS)
        relation_score = min(1.0, max(0.0, relation_count / 3.0))

        # 5. Question type / semantic complexity
        if is_chinese:
            question_type = self._detect_chinese_question_type(q)
        else:
            question_type = self._detect_english_question_type(q)
        semantic_score = self.QUESTION_TYPE_COMPLEXITY.get(question_type, 0.5)

        # 6. Hop demand
        if is_chinese:
            conjunction_count = len(re.findall(r'\b(和|与|或|并且|但是|因此|而且|此外)\b', q))
            question_mark_count = q.count('?') + q.count('？')
        else:
            conjunction_count = len(re.findall(r'\b(and|or|but|however|therefore|moreover|furthermore)\b', q.lower()))
            question_mark_count = q.count('?')
        hop_demand_score = min(1.0, max(0.0, (conjunction_count + question_mark_count - 1) / 3.0))

        # Combined complexity score
        combined = (
            length_score * 0.15 +
            sentence_score * 0.15 +
            entity_score * 0.20 +
            relation_score * 0.20 +
            semantic_score * 0.20 +
            hop_demand_score * 0.10
        )

        return {
            "complexity_score": round(combined, 4),
            "question_type": question_type,
            "features": {
                "length": length,
                "length_score": round(length_score, 4),
                "sentence_count": sentence_count,
                "sentence_score": round(sentence_score, 4),
                "entity_count": entity_count,
                "entity_score": round(entity_score, 4),
                "relation_count": relation_count,
                "relation_score": round(relation_score, 4),
                "semantic_score": round(semantic_score, 4),
                "hop_demand_score": round(hop_demand_score, 4),
            },
        }

    def _extract_chinese_entities(self, query: str) -> tuple[int, List[str]]:
        """Extract Chinese entities using jieba POS tagging."""
        import jieba.posseg as pseg
        entities = []
        for word, flag in pseg.cut(query):
            if flag in ("nr", "ns", "nt", "nz"):  # person, place, org, proper noun
                entities.append(word)
        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique_entities.append(e)
        return len(unique_entities), unique_entities

    def _count_chinese_relation_words(self, query: str) -> int:
        """Count Chinese relation words in query."""
        count = 0
        for word in self.CHINESE_RELATION_WORDS:
            if word in query:
                count += 1
        return count

    def _detect_english_question_type(self, query: str) -> str:
        q = query.lower()
        # Multi-hop: multiple questions or multiple explicit entities with relations
        if q.count('?') > 1 or len(re.findall(r'\b(and|or)\b', q)) > 2:
            return "multi_hop"
        if any(w in q for w in ["compare", "vs", "versus", "difference between", "similarities", "better than", "worse than", "advantage", "disadvantage"]):
            return "comparative"
        if any(w in q for w in ["why", "cause", "reason", "due to", "because", "lead to", "result in", "consequence"]):
            return "causal"
        if any(w in q for w in ["what is", "what are", "define", "definition", "meaning of", "explain"]):
            return "definition"
        if any(w in q for w in ["list", "enumerate", "all the", "examples of", "name some"]):
            return "list"
        return "factual"

    def _detect_chinese_question_type(self, query: str) -> str:
        q = query.lower()
        # Multi-hop: multiple questions or multiple conjunctions
        if q.count('?') + q.count('？') > 1 or len(re.findall(r'\b(和|与|或|并且)\b', q)) > 2:
            return "multi_hop"
        if any(w in q for w in ["比较", "对比", "差异", "区别", "不同", "vs", "versus", "优势", "劣势", "优缺点"]):
            return "comparative"
        if any(w in q for w in ["为什么", "原因", "导致", "引起", "由于", "因为", "结果", "后果"]):
            return "causal"
        if any(w in q for w in ["什么是", "定义", "含义", "意思", "解释", "介绍"]):
            return "definition"
        if any(w in q for w in ["列表", "列举", "所有", "例子", "有哪些", "叫什么"]):
            return "list"
        return "factual"


complexity_analyzer = ComplexityAnalyzer()
