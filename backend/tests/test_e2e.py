"""End-to-end integration tests for RAG pipeline.
Mocks external LLM and retriever calls to avoid network dependencies.
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.cache import query_cache, answer_cache


@pytest_asyncio.fixture(autouse=True)
async def clear_caches():
    """Clear caches before each e2e test."""
    query_cache.clear()
    answer_cache.clear()


def _mock_retriever(chunks):
    """Create a mock retriever that returns predefined chunks."""
    mock = MagicMock()
    mock.retrieve = AsyncMock(return_value={"chunks": chunks, "total_found": len(chunks)})
    return mock


@pytest.mark.asyncio
async def test_pipeline_event_sequence(db_session: AsyncSession):
    """Verify SSE event sequence: step -> done."""
    from app.services.rag_pipeline import run_rag_pipeline

    mock_chunks = [
        {"content": "Paris is the capital of France.", "source": "test", "relevance_score": 0.9, "chunk_index": 0},
    ]
    mock_retriever = _mock_retriever(mock_chunks)

    with patch("app.services.rag_pipeline.retriever_registry.get", return_value=mock_retriever):
        with patch("app.services.rag_pipeline.deepseek_client.generate_answer", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Paris is the capital of France. [ref:chunk_0]"

            events = []
            async for event in run_rag_pipeline(db_session, "What is the capital of France?", strategy="vector"):
                events.append(event)

            event_types = [e["event"] for e in events]
            assert "session_created" in event_types
            assert "step" in event_types
            assert "done" in event_types
            assert event_types[-1] == "done"


@pytest.mark.asyncio
async def test_pipeline_caches_query(db_session: AsyncSession):
    """Second identical query should hit cache."""
    from app.services.rag_pipeline import run_rag_pipeline

    mock_chunks = [
        {"content": "ML is a subset of AI.", "source": "test", "relevance_score": 0.8, "chunk_index": 0},
    ]
    mock_retriever = _mock_retriever(mock_chunks)
    call_count = 0

    def get_retriever(name):
        nonlocal call_count
        call_count += 1
        return mock_retriever

    with patch("app.services.rag_pipeline.retriever_registry.get", side_effect=get_retriever):
        with patch("app.services.rag_pipeline.deepseek_client.generate_answer", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Cached answer. [ref:chunk_0]"

            # First run
            events1 = []
            async for event in run_rag_pipeline(db_session, "What is ML?", strategy="vector"):
                events1.append(event)

            # Second run (should use cached retrieval, retriever called once)
            events2 = []
            async for event in run_rag_pipeline(db_session, "What is ML?", strategy="vector"):
                events2.append(event)

            # Retriever should only be called once due to cache
            assert call_count == 1


@pytest.mark.asyncio
async def test_pipeline_with_hybrid_strategy(db_session: AsyncSession):
    """Hybrid strategy should produce a completed session."""
    from app.services.rag_pipeline import run_rag_pipeline

    mock_chunks = [
        {"content": "Test content.", "source": "test", "relevance_score": 0.7, "chunk_index": 0},
    ]
    mock_retriever = _mock_retriever(mock_chunks)

    with patch("app.services.rag_pipeline.retriever_registry.get", return_value=mock_retriever):
        with patch("app.services.rag_pipeline.deepseek_client.generate_answer", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Answer with [ref:chunk_0]"

            events = []
            async for event in run_rag_pipeline(db_session, "Test query", strategy="hybrid"):
                events.append(event)

            done_events = [e for e in events if e["event"] == "done"]
            assert len(done_events) == 1
            assert done_events[0]["data"]["session_id"] is not None


@pytest.mark.asyncio
async def test_pipeline_answer_segments(db_session: AsyncSession):
    """Answer generation step should include parsed segments."""
    from app.services.rag_pipeline import run_rag_pipeline

    mock_chunks = [
        {"content": "Fact one.", "source": "test", "relevance_score": 0.9, "chunk_index": 0},
        {"content": "Fact two.", "source": "test", "relevance_score": 0.8, "chunk_index": 1},
    ]
    mock_retriever = _mock_retriever(mock_chunks)

    with patch("app.services.rag_pipeline.retriever_registry.get", return_value=mock_retriever):
        with patch("app.services.rag_pipeline.deepseek_client.generate_answer", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "First fact [ref:chunk_0]. Second fact [ref:chunk_1]."

            events = []
            async for event in run_rag_pipeline(db_session, "Test query", strategy="vector"):
                events.append(event)

            # Find answer_generate step
            answer_steps = [e for e in events if e["event"] == "step" and e["data"].get("step_type") == "answer_generate"]
            assert len(answer_steps) == 1
            output = answer_steps[0]["data"]["output_data"]
            assert "answer_segments" in output
            segments = output["answer_segments"]
            assert len(segments) >= 2
            # At least one segment should have support
            supported = [s for s in segments if s["supported_by"]]
            assert len(supported) >= 1
