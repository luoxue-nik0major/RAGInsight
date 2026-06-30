from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List, Dict, Any
import asyncio

from app.core.database import get_db, AsyncSessionLocal
from app.core.models import Session as DBSession, Step, StepType, Chunk, AttributionResult
from app.schemas import (
    PerturbationResult, WhatIfRequest, WhatIfResponse,
    AttributionReportSchema, InterventionResultSchema, CausalGraph,
)
from app.services.perturbation import PerturbationAnalyzer, task_manager
from app.services.causal_attribution import causal_analyzer

router = APIRouter(prefix="/api", tags=["perturbation"])


async def _run_perturbation_background(
    task_id: str,
    session_id: int,
    query: str,
    answer: str,
    chunks: List[Dict[str, Any]],
):
    """Background coroutine for perturbation analysis."""
    try:
        task_manager.update_task(task_id, status="running", progress=0, total=len(chunks))

        def on_progress(current: int, total: int):
            task_manager.update_task(task_id, progress=current, total=total)

        analyzer = PerturbationAnalyzer()
        results = await analyzer.analyze(query, answer, chunks, on_progress=on_progress)

        # Update importance_score in database
        async with AsyncSessionLocal() as db:
            for r in results:
                cid = r.get("chunk_id")
                if cid is not None:
                    await db.execute(
                        update(Chunk)
                        .where(Chunk.id == cid)
                        .values(importance_score=r["importance_score"])
                    )
            await db.commit()

        task_manager.update_task(
            task_id,
            status="completed",
            progress=len(chunks),
            result=results,
        )
    except Exception as e:
        task_manager.update_task(task_id, status="failed", error=str(e))


@router.post("/sessions/{session_id}/perturbation")
async def trigger_perturbation(
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Trigger full perturbation analysis for a session."""
    result = await db.execute(select(DBSession).where(DBSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.final_answer:
        raise HTTPException(status_code=400, detail="Session has no final answer")

    # Find vector retrieve step
    steps_result = await db.execute(
        select(Step).where(Step.session_id == session_id)
    )
    steps = steps_result.scalars().all()

    retrieve_step = None
    for step in steps:
        if step.step_type == StepType.VECTOR_RETRIEVE:
            retrieve_step = step
            break

    if not retrieve_step:
        raise HTTPException(status_code=400, detail="No retrieve step found")

    chunks_result = await db.execute(select(Chunk).where(Chunk.step_id == retrieve_step.id))
    chunks = chunks_result.scalars().all()

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found")

    chunk_dicts = [
        {
            "content": c.content,
            "source": c.source,
            "relevance_score": c.relevance_score,
            "chunk_index": c.chunk_index,
            "id": c.id,
        }
        for c in chunks
    ]

    task_id = await task_manager.create_task(session_id)
    task_manager.update_task(task_id, total=len(chunks))

    # Fire-and-forget background task
    asyncio.create_task(
        _run_perturbation_background(
            task_id, session_id, session.query, session.final_answer, chunk_dicts
        )
    )

    return {"task_id": task_id, "status": "pending"}


@router.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get perturbation task status and results."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/sessions/{session_id}/perturbation/what-if", response_model=WhatIfResponse)
async def what_if_perturbation(
    session_id: int,
    body: WhatIfRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate answer after hypothetically removing selected chunks."""
    result = await db.execute(select(DBSession).where(DBSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.final_answer:
        raise HTTPException(status_code=400, detail="Session has no final answer")

    # Find retrieve step and chunks
    steps_result = await db.execute(
        select(Step).where(Step.session_id == session_id)
    )
    steps = steps_result.scalars().all()

    retrieve_step = None
    for step in steps:
        if step.step_type == StepType.VECTOR_RETRIEVE:
            retrieve_step = step
            break

    if not retrieve_step:
        raise HTTPException(status_code=400, detail="No retrieve step found")

    chunks_result = await db.execute(select(Chunk).where(Chunk.step_id == retrieve_step.id))
    chunks = chunks_result.scalars().all()

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found")

    chunk_dicts = [
        {
            "content": c.content,
            "source": c.source,
            "relevance_score": c.relevance_score,
            "chunk_index": c.chunk_index,
            "id": c.id,
        }
        for c in chunks
    ]

    analyzer = PerturbationAnalyzer()
    what_if_result = await analyzer.what_if(
        session.query,
        session.final_answer,
        chunk_dicts,
        body.remove_chunk_indices,
    )

    return WhatIfResponse(
        original_answer=what_if_result["original_answer"],
        new_answer=what_if_result["new_answer"],
        similarity=what_if_result["similarity"],
        removed_count=what_if_result["removed_count"],
        kept_count=what_if_result["kept_count"],
    )


# ── Causal Attribution Endpoints ────────────────────────────────────────

async def _run_attribution_background(
    task_id: str,
    session_id: int,
    query: str,
    answer: str,
    chunks: List[Dict[str, Any]],
    strategy: str,
    topk: int,
    trace: Dict[str, Any],
):
    """Background coroutine for causal attribution analysis."""
    try:
        task_manager.update_task(task_id, status="running", progress=0, total=100)

        def on_progress(stage: str, current: int, total: int):
            pct = 0
            if stage == "approximate_done":
                pct = 30
            elif stage == "llm_intervention":
                pct = 30 + int((current / max(total, 1)) * 60)
            task_manager.update_task(task_id, progress=pct, total=100,
                                     extra_info={"stage": stage, "current": current, "total": total})

        report = await causal_analyzer.run_full_attribution(
            query=query,
            original_answer=answer,
            chunks=chunks,
            original_strategy=strategy,
            original_topk=topk,
            session_trace=trace,
            on_progress=on_progress,
        )
        report.session_id = session_id

        # Persist attribution results to DB
        async with AsyncSessionLocal() as db:
            for intervention in report.interventions:
                db.add(AttributionResult(
                    session_id=session_id,
                    component_name=intervention.component.value,
                    intervention_type=intervention.intervention.value,
                    intervention_params=intervention.params,
                    original_quality=intervention.original_quality,
                    perturbed_quality=intervention.perturbed_quality,
                    quality_delta=intervention.quality_delta,
                    attribution_score=intervention.attribution_score,
                    is_approximate=1 if intervention.is_approximate else 0,
                    description=intervention.description,
                ))
            await db.commit()

        # Serialize for task result
        result_dict = {
            "session_id": report.session_id,
            "query": report.query,
            "original_strategy": report.original_strategy,
            "original_quality": report.original_quality,
            "interventions": [
                {
                    "component": r.component.value,
                    "intervention": r.intervention.value,
                    "params": r.params,
                    "original_quality": r.original_quality,
                    "perturbed_quality": r.perturbed_quality,
                    "quality_delta": r.quality_delta,
                    "attribution_score": r.attribution_score,
                    "is_approximate": r.is_approximate,
                    "description": r.description,
                }
                for r in report.interventions
            ],
            "component_attributions": report.component_attributions,
            "top_contributors": report.top_contributors,
            "causal_graph": report.causal_graph,
            "total_interventions": report.total_interventions,
            "llm_interventions": report.llm_interventions,
            "duration_ms": report.duration_ms,
        }
        task_manager.update_task(task_id, status="completed", progress=100, result=result_dict)
    except Exception as e:
        task_manager.update_task(task_id, status="failed", error=str(e))


@router.post("/sessions/{session_id}/causal-attribution")
async def trigger_causal_attribution(
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Trigger full causal attribution analysis for a session."""
    result = await db.execute(select(DBSession).where(DBSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.final_answer:
        raise HTTPException(status_code=400, detail="Session has no final answer")

    steps_result = await db.execute(
        select(Step).where(Step.session_id == session_id)
    )
    steps = steps_result.scalars().all()

    retrieve_step = None
    for step in steps:
        if step.step_type == StepType.VECTOR_RETRIEVE:
            retrieve_step = step
            break

    if not retrieve_step:
        raise HTTPException(status_code=400, detail="No retrieve step found")

    chunks_result = await db.execute(select(Chunk).where(Chunk.step_id == retrieve_step.id))
    chunks = chunks_result.scalars().all()

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found")

    chunk_dicts = [
        {
            "content": c.content,
            "source": c.source,
            "relevance_score": c.relevance_score,
            "chunk_index": c.chunk_index,
            "id": c.id,
        }
        for c in chunks
    ]

    # Determine strategy and top_k from step input_data
    strategy = retrieve_step.input_data.get("strategy", "vector") if retrieve_step.input_data else "vector"
    topk = retrieve_step.input_data.get("top_k", 5) if retrieve_step.input_data else 5
    trace = session.execution_trace or {}

    task_id = await task_manager.create_task(session_id)
    task_manager.update_task(task_id, total=len(chunks), status="pending")

    asyncio.create_task(
        _run_attribution_background(
            task_id, session_id, session.query, session.final_answer,
            chunk_dicts, strategy, topk, trace,
        )
    )

    return {"task_id": task_id, "status": "pending", "message": "Causal attribution analysis started"}


@router.get("/sessions/{session_id}/causal-attribution")
async def get_causal_attribution(
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get saved causal attribution results for a session."""
    result = await db.execute(
        select(AttributionResult).where(AttributionResult.session_id == session_id)
    )
    rows = result.scalars().all()

    if not rows:
        raise HTTPException(status_code=404, detail="No attribution results found for this session")

    interventions = []
    for r in rows:
        interventions.append(InterventionResultSchema(
            component=r.component_name,
            intervention=r.intervention_type,
            params=r.intervention_params or {},
            original_quality=r.original_quality,
            perturbed_quality=r.perturbed_quality,
            quality_delta=r.quality_delta,
            attribution_score=r.attribution_score,
            is_approximate=bool(r.is_approximate),
            description=r.description or "",
        ))

    # Aggregate component attributions
    comp_scores: Dict[str, float] = {}
    for r in rows:
        key = r.component_name
        comp_scores[key] = comp_scores.get(key, 0.0) + r.attribution_score
    total = sum(comp_scores.values())
    if total > 0:
        for k in comp_scores:
            comp_scores[k] = round(comp_scores[k] / total, 4)

    return {
        "session_id": session_id,
        "interventions": [i.model_dump() for i in interventions],
        "component_attributions": comp_scores,
        "total_results": len(rows),
    }
