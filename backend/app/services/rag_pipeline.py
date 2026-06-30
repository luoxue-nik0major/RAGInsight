import time
import re
from typing import List, Dict, Any, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models import Session as DBSession, Step, StepType, Chunk, SessionStatus, Alert, AlertSeverity
from app.services.retriever import retriever_registry
from app.services.deepseek import deepseek_client
from app.services.quality import quality_evaluator
from app.services.root_cause import root_cause_analyzer
from app.services.complexity import complexity_analyzer
from app.services.strategy_recommender import strategy_recommender
from app.services.cache import query_cache, answer_cache
from app.services.answer_evaluator import answer_evaluator

# Context length threshold (characters, approximate for Chinese/English mix)
CONTEXT_LENGTH_WARNING = 3000
CONTEXT_LENGTH_ERROR = 6000


async def run_rag_pipeline(
    db: AsyncSession,
    query: str,
    strategy: str = "vector",
    collection: str = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Run full RAG pipeline and yield SSE events.
    Events: step | alert | root_cause | done | error
    """
    start_time = time.time()
    all_steps_data: List[Dict[str, Any]] = []
    all_alerts_data: List[Dict[str, Any]] = []
    
    # 1. Analyze complexity & recommend strategy
    complexity_result = complexity_analyzer.analyze(query)
    recommendation = strategy_recommender.recommend(
        complexity_result["complexity_score"],
        query,
        {**complexity_result["features"], "question_type": complexity_result["question_type"]},
    )

    # Create session
    session = DBSession(
        query=query,
        status=SessionStatus.RUNNING,
        complexity_score=complexity_result["complexity_score"],
        recommended_strategy=recommendation["recommended_strategy"],
        execution_trace={"start_time": start_time, "complexity": complexity_result, "recommendation": recommendation, "actual_strategy": strategy},
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    yield {"event": "session_created", "data": {"session_id": session.id, "query": query, "complexity": complexity_result, "recommendation": recommendation}}

    try:
        # 2. Query Parse Step
        step_start = time.time()
        entities = list(set(re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', query)))
        parse_output = {
            "entities": entities,
            "question_type": "factual",
            "keywords": query.split(),
        }
        parse_step = Step(
            session_id=session.id,
            step_type=StepType.QUERY_PARSE,
            input_data={"query": query},
            output_data=parse_output,
            duration_ms=int((time.time() - step_start) * 1000),
        )
        db.add(parse_step)
        await db.commit()
        await db.refresh(parse_step)
        step_dict = step_to_dict(parse_step)
        all_steps_data.append(step_dict)
        yield {"event": "step", "data": step_dict}

        # 3. Vector Retrieve Step
        step_start = time.time()

        # Check query cache
        cached_result = query_cache.get(query, strategy)
        if cached_result:
            retrieve_result = cached_result
        else:
            retriever = retriever_registry.get(strategy)
            retrieve_result = await retriever.retrieve(query, top_k=5, collection_name=collection)
            query_cache.set(query, strategy, retrieve_result)
        
        # Quality evaluation
        quality = quality_evaluator.evaluate_all(query, retrieve_result["chunks"])

        retrieve_step = Step(
            session_id=session.id,
            step_type=StepType.VECTOR_RETRIEVE,
            input_data={"query": query, "strategy": strategy, "top_k": 5},
            output_data={"total_found": retrieve_result["total_found"], "quality_metrics": quality},
            quality_score=quality["combined"],
            duration_ms=int((time.time() - step_start) * 1000),
        )
        db.add(retrieve_step)
        await db.commit()
        await db.refresh(retrieve_step)

        # Add chunks
        for chunk_data in retrieve_result["chunks"]:
            chunk = Chunk(
                step_id=retrieve_step.id,
                content=chunk_data["content"],
                source=chunk_data.get("source", ""),
                relevance_score=chunk_data["relevance_score"],
                chunk_index=chunk_data["chunk_index"],
            )
            db.add(chunk)
        await db.commit()
        await db.refresh(retrieve_step)
        step_dict = step_to_dict(retrieve_step, chunks=retrieve_result["chunks"])
        all_steps_data.append(step_dict)
        yield {"event": "step", "data": step_dict}

        # Alerts for retrieval step
        if not retrieve_result["chunks"]:
            alert = Alert(
                session_id=session.id,
                step_id=retrieve_step.id,
                alert_type="empty_results",
                severity=AlertSeverity.ERROR,
                message="检索结果为空，知识库中可能没有相关信息。",
                suggestion="尝试扩展查询关键词或使用不同的检索策略。",
            )
            db.add(alert)
            await db.commit()
            alert_dict = alert_to_dict(alert)
            all_alerts_data.append(alert_dict)
            yield {"event": "alert", "data": alert_dict}
        else:
            if quality["relevance"] < 0.3:
                alert = Alert(
                    session_id=session.id,
                    step_id=retrieve_step.id,
                    alert_type="low_relevance",
                    severity=AlertSeverity.WARNING,
                    message=f"检索结果相关性较低（平均分数: {quality['relevance']:.2f}）。",
                    suggestion="检查查询是否与知识库主题匹配，或尝试增加检索数量。",
                )
                db.add(alert)
                await db.commit()
                alert_dict = alert_to_dict(alert)
                all_alerts_data.append(alert_dict)
                yield {"event": "alert", "data": alert_dict}

            if quality["coverage"] < 0.3:
                alert = Alert(
                    session_id=session.id,
                    step_id=retrieve_step.id,
                    alert_type="low_coverage",
                    severity=AlertSeverity.WARNING,
                    message=f"检索结果未充分覆盖查询关键实体（覆盖率: {quality['coverage']:.0%}）。",
                    suggestion="查询中的关键概念在检索结果中未完全出现，尝试改写查询或使用混合检索。",
                )
                db.add(alert)
                await db.commit()
                alert_dict = alert_to_dict(alert)
                all_alerts_data.append(alert_dict)
                yield {"event": "alert", "data": alert_dict}

            if quality["diversity"] < 0.2:
                alert = Alert(
                    session_id=session.id,
                    step_id=retrieve_step.id,
                    alert_type="low_diversity",
                    severity=AlertSeverity.WARNING,
                    message=f"检索结果多样性不足（多样性分数: {quality['diversity']:.2f}），多个片段内容高度相似。",
                    suggestion="检索结果可能来自同一文档的相似段落，建议增加检索来源的多样性。",
                )
                db.add(alert)
                await db.commit()
                alert_dict = alert_to_dict(alert)
                all_alerts_data.append(alert_dict)
                yield {"event": "alert", "data": alert_dict}

        # 4. Context Build Step
        step_start = time.time()
        context_chunks = retrieve_result["chunks"]
        context_text = "\n\n".join(c["content"] for c in context_chunks)
        
        context_quality = 1.0
        if len(context_text) > CONTEXT_LENGTH_ERROR:
            context_quality = 0.0
        elif len(context_text) > CONTEXT_LENGTH_WARNING:
            context_quality = 0.5

        context_step = Step(
            session_id=session.id,
            step_type=StepType.CONTEXT_BUILD,
            input_data={"chunk_count": len(context_chunks)},
            output_data={"context_length": len(context_text), "chunk_ids": [c["chunk_index"] for c in context_chunks]},
            quality_score=context_quality,
            duration_ms=int((time.time() - step_start) * 1000),
        )
        db.add(context_step)
        await db.commit()
        await db.refresh(context_step)
        step_dict = step_to_dict(context_step)
        all_steps_data.append(step_dict)
        yield {"event": "step", "data": step_dict}

        if len(context_text) > CONTEXT_LENGTH_WARNING:
            is_error = len(context_text) > CONTEXT_LENGTH_ERROR
            alert = Alert(
                session_id=session.id,
                step_id=context_step.id,
                alert_type="context_too_long",
                severity=AlertSeverity.ERROR if is_error else AlertSeverity.WARNING,
                message=f"上下文长度{'超过限制' if is_error else '接近限制'}（{len(context_text)} 字符）。",
                suggestion="减少检索片段数量或对片段进行摘要压缩。",
            )
            db.add(alert)
            await db.commit()
            alert_dict = alert_to_dict(alert)
            all_alerts_data.append(alert_dict)
            yield {"event": "alert", "data": alert_dict}

        # 5. Answer Generate Step
        step_start = time.time()

        # Check answer cache for simple factual queries
        cached_answer = None
        if complexity_result["complexity_score"] < 0.3:
            cached_answer = answer_cache.get(query, strategy)

        if cached_answer:
            answer = cached_answer
        elif context_chunks:
            answer = await deepseek_client.generate_answer(query, context_chunks)
            # Cache answer for simple factual queries
            if complexity_result["complexity_score"] < 0.3:
                answer_cache.set(query, strategy, answer)
        else:
            answer = "抱歉，未能在知识库中找到相关信息来回答您的问题。"

        citations = []
        for match in re.finditer(r'\[ref:chunk_(\d+)\]', answer):
            citations.append(int(match.group(1)))

        valid_chunk_indices = set(c["chunk_index"] for c in context_chunks)
        invalid_citations = [c for c in citations if c not in valid_chunk_indices]

        answer_quality = 1.0
        if invalid_citations:
            answer_quality = 0.3
        elif not citations and context_chunks:
            answer_quality = 0.7

        # Parse answer into segments with citation support info
        answer_segments = _parse_answer_segments(answer, valid_chunk_indices)

        answer_step = Step(
            session_id=session.id,
            step_type=StepType.ANSWER_GENERATE,
            input_data={"context_length": len(context_text)},
            output_data={"answer": answer, "citations": citations, "invalid_citations": invalid_citations, "answer_segments": answer_segments},
            quality_score=answer_quality,
            duration_ms=int((time.time() - step_start) * 1000),
        )
        db.add(answer_step)
        await db.commit()
        await db.refresh(answer_step)
        step_dict = step_to_dict(answer_step)
        all_steps_data.append(step_dict)
        yield {"event": "step", "data": step_dict}

        if invalid_citations:
            alert = Alert(
                session_id=session.id,
                step_id=answer_step.id,
                alert_type="hallucination_risk",
                severity=AlertSeverity.WARNING,
                message=f"答案引用了不存在的检索片段（chunk_{', chunk_'.join(map(str, invalid_citations))}），存在幻觉风险。",
                suggestion="检查模型输出，确保引用标记与检索结果对应。",
            )
            db.add(alert)
            await db.commit()
            alert_dict = alert_to_dict(alert)
            all_alerts_data.append(alert_dict)
            yield {"event": "alert", "data": alert_dict}

        # 6. Answer Evaluation (LLM-as-a-Judge, local-only)
        answer_eval = answer_evaluator.evaluate(query, answer, context_chunks) if context_chunks else None

        # 7. Root Cause Analysis
        root_cause = root_cause_analyzer.analyze(all_steps_data, all_alerts_data)
        if root_cause:
            yield {"event": "root_cause", "data": root_cause}

        # Update session
        session.status = SessionStatus.COMPLETED
        session.final_answer = answer
        session.execution_trace = {
            **session.execution_trace,
            "total_duration_ms": int((time.time() - start_time) * 1000),
            "steps_count": 4,
            "quality_metrics": quality if retrieve_result["chunks"] else {},
            "root_cause": root_cause,
            "answer_evaluation": answer_eval,
        }
        await db.commit()

        yield {"event": "done", "data": {"session_id": session.id, "answer": answer, "answer_evaluation": answer_eval}}

    except Exception as e:
        session.status = SessionStatus.FAILED
        session.execution_trace = {
            **session.execution_trace,
            "error": str(e),
        }
        await db.commit()
        yield {"event": "error", "data": {"message": str(e)}}


def step_to_dict(step: Step, chunks: List[Dict] = None) -> Dict[str, Any]:
    result = {
        "id": step.id,
        "session_id": step.session_id,
        "step_type": step.step_type.value,
        "input_data": step.input_data,
        "output_data": step.output_data,
        "quality_score": step.quality_score,
        "duration_ms": step.duration_ms,
        "timestamp": step.timestamp.isoformat() if step.timestamp else None,
        "chunks": chunks or [],
    }
    return result


def alert_to_dict(alert: Alert) -> Dict[str, Any]:
    return {
        "id": alert.id,
        "session_id": alert.session_id,
        "step_id": alert.step_id,
        "alert_type": alert.alert_type,
        "severity": alert.severity.value,
        "message": alert.message,
        "suggestion": alert.suggestion,
        "created_at": alert.created_at.isoformat() if alert.created_at else None,
    }


def _parse_answer_segments(answer: str, valid_chunk_indices: set) -> List[Dict[str, Any]]:
    """
    Split answer into segments by citation markers.
    Each segment is annotated with the chunk indices that support it.
    """
    segments = []
    # Pattern to split by citation markers while keeping them
    parts = re.split(r'(\[ref:chunk_\d+\])', answer)

    current_support: List[int] = []
    for part in parts:
        if not part:
            continue
        m = re.match(r'\[ref:chunk_(\d+)\]', part)
        if m:
            idx = int(m.group(1))
            if idx in valid_chunk_indices:
                current_support = [idx]
            else:
                current_support = []
        else:
            segments.append({
                "text": part,
                "supported_by": current_support.copy(),
            })
            # Reset support for next segment (support only applies to preceding text)
            current_support = []
    return segments
