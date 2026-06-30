from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    """Run database migrations on startup."""
    from alembic.config import Config as AlembicConfig
    from alembic import command
    import os

    alembic_ini = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "alembic.ini",
    )
    alembic_cfg = AlembicConfig(alembic_ini)
    # Run migrations in-process using an async approach:
    # For SQLite we use create_all as a simpler alternative to async migration runner
    # In production, run `alembic upgrade head` separately
    from app.core.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
