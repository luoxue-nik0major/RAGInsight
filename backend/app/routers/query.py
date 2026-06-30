from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator
import json

from app.core.database import get_db
from app.services.rag_pipeline import run_rag_pipeline

router = APIRouter(prefix="/api", tags=["query"])


async def sse_stream(
    db: AsyncSession,
    query: str,
    strategy: str = "vector",
    collection: str = None,
) -> AsyncGenerator[str, None]:
    async for event in run_rag_pipeline(db, query, strategy, collection):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/query")
async def post_query(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Execute RAG query and stream results via SSE."""
    body = await request.json()
    query = body.get("query", "").strip()
    strategy = body.get("strategy", "vector")
    collection = body.get("collection", None)

    if not query:
        return StreamingResponse(
            (f'data: {json.dumps({"event": "error", "data": {"message": "Query cannot be empty"}})}\n\n' for _ in range(1)),
            media_type="text/event-stream",
        )

    return StreamingResponse(
        sse_stream(db, query, strategy, collection),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
