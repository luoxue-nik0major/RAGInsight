"""
Export service for Phase 5.
Serializes session data to structured JSON.
"""
from typing import Dict, Any, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.core.models import Session as DBSession, Step, Chunk, Alert


async def export_session_json(session_id: int) -> Optional[Dict[str, Any]]:
    """Export a complete session with steps, chunks, and alerts as JSON."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(DBSession).where(DBSession.id == session_id))
        session = result.scalar_one_or_none()
        if not session:
            return None

        # Load steps
        steps_result = await db.execute(select(Step).where(Step.session_id == session_id).order_by(Step.timestamp))
        steps = steps_result.scalars().all()

        step_data = []
        for step in steps:
            chunks_result = await db.execute(select(Chunk).where(Chunk.step_id == step.id))
            chunks = [
                {
                    "id": c.id,
                    "content": c.content,
                    "source": c.source,
                    "relevance_score": c.relevance_score,
                    "importance_score": c.importance_score,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks_result.scalars().all()
            ]

            alerts_result = await db.execute(select(Alert).where(Alert.step_id == step.id))
            alerts = [
                {
                    "id": a.id,
                    "alert_type": a.alert_type,
                    "severity": a.severity.value,
                    "message": a.message,
                    "suggestion": a.suggestion,
                }
                for a in alerts_result.scalars().all()
            ]

            step_data.append({
                "id": step.id,
                "step_type": step.step_type.value,
                "input_data": step.input_data,
                "output_data": step.output_data,
                "quality_score": step.quality_score,
                "duration_ms": step.duration_ms,
                "timestamp": step.timestamp.isoformat() if step.timestamp else None,
                "chunks": chunks,
                "alerts": alerts,
            })

        # Session-level alerts
        session_alerts_result = await db.execute(
            select(Alert).where(Alert.session_id == session_id, Alert.step_id.is_(None))
        )
        session_alerts = [
            {
                "id": a.id,
                "alert_type": a.alert_type,
                "severity": a.severity.value,
                "message": a.message,
                "suggestion": a.suggestion,
            }
            for a in session_alerts_result.scalars().all()
        ]

        return {
            "session": {
                "id": session.id,
                "query": session.query,
                "final_answer": session.final_answer,
                "complexity_score": session.complexity_score,
                "recommended_strategy": session.recommended_strategy,
                "status": session.status.value if session.status else None,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                "execution_trace": session.execution_trace,
            },
            "steps": step_data,
            "session_alerts": session_alerts,
        }
