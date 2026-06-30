"""
Experiment runner for Phase 5.
Batch executes RAG pipeline over test queries and persists results.
"""
import asyncio
import json
import time
import os
from typing import List, Dict, Any
from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.core.models import Session as DBSession, Step, Chunk, Alert
from app.services.rag_pipeline import run_rag_pipeline

EXPERIMENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments",
)


class ExperimentRunner:
    """Run batch experiments over test queries."""

    def __init__(self, max_concurrent: int = 2):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False
        self._progress: Dict[str, Any] = {"current": 0, "total": 0, "status": "idle"}
        self._latest_results: List[Dict[str, Any]] = []

    async def run_batch(
        self,
        queries: List[Dict[str, Any]],
        strategies: List[str] = None,
    ) -> List[Dict[str, Any]]:
        strategies = strategies or ["vector"]
        self._running = True
        total = len(queries) * len(strategies)
        self._progress = {"current": 0, "total": total, "status": "running"}
        self._latest_results = []

        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        filepath = os.path.join(EXPERIMENT_DIR, f"experiment_{int(time.time())}.jsonl")

        for query_item in queries:
            for strategy in strategies:
                async with self.semaphore:
                    result = await self._run_single(query_item, strategy)
                    self._latest_results.append(result)
                    self._progress["current"] += 1
                    # Append to file incrementally
                    with open(filepath, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

        self._progress["status"] = "completed"
        self._running = False
        return self._latest_results

    async def _run_single(self, query_item: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        async with AsyncSessionLocal() as db:
            try:
                session_id = None
                async for event in run_rag_pipeline(db, query_item["query"], strategy):
                    if event["event"] == "done":
                        session_id = event["data"]["session_id"]
                    elif event["event"] == "error":
                        return {
                            "query_id": query_item.get("id"),
                            "query": query_item["query"],
                            "strategy": strategy,
                            "category": query_item.get("category"),
                            "expected_alerts": query_item.get("expected_alerts", []),
                            "status": "failed",
                            "error": event["data"].get("message", "Unknown error"),
                        }

                if session_id is None:
                    raise Exception("Pipeline did not return session_id")

                # Load full session
                result = await db.execute(select(DBSession).where(DBSession.id == session_id))
                session = result.scalar_one()

                # Load steps
                steps_result = await db.execute(select(Step).where(Step.session_id == session_id))
                steps = steps_result.scalars().all()

                step_data = []
                for step in steps:
                    chunks_result = await db.execute(select(Chunk).where(Chunk.step_id == step.id))
                    chunks = [
                        {
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
                            "alert_type": a.alert_type,
                            "severity": a.severity.value,
                            "message": a.message,
                        }
                        for a in alerts_result.scalars().all()
                    ]

                    step_data.append({
                        "step_type": step.step_type.value,
                        "input_data": step.input_data,
                        "output_data": step.output_data,
                        "quality_score": step.quality_score,
                        "duration_ms": step.duration_ms,
                        "chunks": chunks,
                        "alerts": alerts,
                    })

                # Collect all alert types
                all_alert_types = []
                for s in step_data:
                    all_alert_types.extend([a["alert_type"] for a in s["alerts"]])

                return {
                    "query_id": query_item.get("id"),
                    "query": query_item["query"],
                    "strategy": strategy,
                    "category": query_item.get("category"),
                    "expected_alerts": query_item.get("expected_alerts", []),
                    "session_id": session_id,
                    "status": session.status.value if session.status else "unknown",
                    "complexity_score": session.complexity_score,
                    "recommended_strategy": session.recommended_strategy,
                    "final_answer": session.final_answer,
                    "actual_alerts": list(set(all_alert_types)),
                    "steps": step_data,
                    "execution_trace": session.execution_trace,
                }
            except Exception as e:
                return {
                    "query_id": query_item.get("id"),
                    "query": query_item["query"],
                    "strategy": strategy,
                    "category": query_item.get("category"),
                    "expected_alerts": query_item.get("expected_alerts", []),
                    "status": "failed",
                    "error": str(e),
                }

    def get_progress(self) -> Dict[str, Any]:
        return self._progress.copy()

    def get_latest_results(self) -> List[Dict[str, Any]]:
        return self._latest_results.copy()

    def is_running(self) -> bool:
        return self._running


experiment_runner = ExperimentRunner()
