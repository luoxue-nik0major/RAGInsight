from fastapi import APIRouter, HTTPException, Request

from app.services.complexity import complexity_analyzer
from app.services.strategy_recommender import get_recommender

router = APIRouter(prefix="/api", tags=["complexity"])


@router.post("/analyze-complexity")
async def analyze_complexity(request: Request):
    """Analyze query complexity and recommend strategy."""
    body = await request.json()
    query = body.get("query", "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    complexity = complexity_analyzer.analyze(query)
    recommendation = get_recommender().recommend(
        complexity["complexity_score"],
        query,
        {**complexity["features"], "question_type": complexity["question_type"]},
    )

    return {
        "query": query,
        **complexity,
        **recommendation,
    }
