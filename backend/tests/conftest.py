"""Pytest fixtures for RAGInsight backend tests."""
import os
import sys
from typing import AsyncGenerator

import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# If the embedding model is already in the local HF cache, skip network
# reachability checks (huggingface.co may be unreachable or slow). On a fresh
# machine without the cache, normal online download still applies.
_hf_model_cache = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--BAAI--bge-small-zh-v1.5",
)
if os.path.exists(_hf_model_cache):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Ensure backend is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.models import Base


# Use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
async def engine():
    """Create a test database engine."""
    eng = create_async_engine(TEST_DATABASE_URL, echo=False, future=True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def db_session(engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide an async database session for each test."""
    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session
        # Rollback after each test to keep isolation
        await session.rollback()
