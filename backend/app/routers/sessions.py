from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from starlette.concurrency import run_in_threadpool
from typing import Dict, List

from app.core.database import get_db
from app.core.models import Session as DBSession, Step, Chunk, Alert
from app.schemas import SessionOut, SessionListItem, StrategyInfo, StepOut, ChunkOut, AlertOut
from app.services.retriever import list_available_collections

router = APIRouter(prefix="/api", tags=["sessions"])


@router.get("/sessions", response_model=List[SessionListItem])
async def list_sessions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(DBSession).order_by(desc(DBSession.created_at)).offset(offset).limit(limit)
    )
    sessions = result.scalars().all()
    return sessions


@router.get("/sessions/{session_id}", response_model=SessionOut)
async def get_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(DBSession).where(DBSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Load steps
    steps_result = await db.execute(
        select(Step).where(Step.session_id == session_id).order_by(Step.timestamp)
    )
    steps = steps_result.scalars().all()
    
    # Batch-load chunks and alerts for all steps (avoid N+1 queries)
    step_ids = [s.id for s in steps]

    chunks_by_step: Dict[int, list] = {}
    alerts_by_step: Dict[int, list] = {}
    if step_ids:
        chunks_result = await db.execute(
            select(Chunk).where(Chunk.step_id.in_(step_ids))
        )
        for c in chunks_result.scalars().all():
            chunks_by_step.setdefault(c.step_id, []).append(ChunkOut.model_validate(c))

        step_alerts_result = await db.execute(
            select(Alert).where(Alert.step_id.in_(step_ids))
        )
        for a in step_alerts_result.scalars().all():
            alerts_by_step.setdefault(a.step_id, []).append(AlertOut.model_validate(a))

    # Build step outputs
    step_outs = []
    for step in steps:
        step_outs.append(StepOut(
            id=step.id,
            session_id=step.session_id,
            step_type=step.step_type,
            input_data=step.input_data or {},
            output_data=step.output_data or {},
            quality_score=step.quality_score,
            duration_ms=step.duration_ms,
            timestamp=step.timestamp,
            chunks=chunks_by_step.get(step.id, []),
            alerts=alerts_by_step.get(step.id, []),
        ))
    
    # Load session-level alerts
    alerts_result = await db.execute(
        select(Alert).where(Alert.session_id == session_id, Alert.step_id.is_(None))
    )
    session_alerts = [AlertOut.model_validate(a) for a in alerts_result.scalars().all()]
    
    return SessionOut(
        id=session.id,
        query=session.query,
        final_answer=session.final_answer,
        complexity_score=session.complexity_score,
        recommended_strategy=session.recommended_strategy,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
        execution_trace=session.execution_trace or {},
        steps=step_outs,
        alerts=session_alerts,
    )


@router.get("/strategies", response_model=List[StrategyInfo])
async def list_strategies():
    return [
        StrategyInfo(
            id="vector",
            name="向量检索",
            description="基于语义相似度的向量检索，适合概念性、语义丰富的查询。",
            icon="search",
        ),
        StrategyInfo(
            id="hybrid",
            name="混合检索",
            description="向量检索 + 关键词检索融合，适合需要精确匹配和语义理解的查询。",
            icon="git-merge",
        ),
        StrategyInfo(
            id="graph",
            name="图检索",
            description="基于知识图谱的多跳推理检索，适合复杂关系查询。",
            icon="share-2",
        ),
    ]


@router.get("/collections")
async def list_collections():
    """List available ChromaDB collections."""
    collections = await run_in_threadpool(list_available_collections)
    return {"collections": collections, "count": len(collections)}
