"""
Database engine, session management, and initialization.
Supports both async (for API) and sync (for background tasks) sessions.
Engines are created lazily to avoid import errors when using MongoDB-only mode.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import structlog
from app.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── Lazy engine references ──
_async_engine = None
_sync_engine = None
_AsyncSessionLocal = None
_SyncSessionLocal = None


def _get_async_engine():
    global _async_engine
    if _async_engine is None:
        from sqlalchemy.ext.asyncio import create_async_engine
        _async_engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
    return _async_engine


def _get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        from sqlalchemy import create_engine
        _sync_engine = create_engine(
            settings.database_url_sync,
            echo=settings.debug,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _sync_engine


def _get_async_session_local():
    global _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
        _AsyncSessionLocal = async_sessionmaker(
            bind=_get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _AsyncSessionLocal


def _get_sync_session_local():
    global _SyncSessionLocal
    if _SyncSessionLocal is None:
        from sqlalchemy.orm import sessionmaker
        _SyncSessionLocal = sessionmaker(bind=_get_sync_engine(), expire_on_commit=False)
    return _SyncSessionLocal


# ── Dependency Injection ──

async def get_async_session():
    """FastAPI dependency for async DB sessions (PostgreSQL only)."""
    if settings.database_type != "postgres":
        yield None
        return

    AsyncSessionLocal = _get_async_session_local()
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_session():
    """Get a sync session for background tasks."""
    return _get_sync_session_local()()


# ── Database Initialization ──

async def init_database():
    """Create all tables if they don't exist (PostgreSQL only)."""
    if settings.database_type != "postgres":
        logger.info("postgres_skipped", reason="using_mongodb")
        return

    from app.db.models import Base
    from sqlalchemy.ext.asyncio import AsyncSession
    engine = _get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("postgres_tables_created")


async def check_db_connection() -> bool:
    """Check if PostgreSQL is reachable."""
    try:
        from sqlalchemy import text
        engine = _get_async_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def drop_all_tables():
    """Drop all tables (use with caution)."""
    from app.db.models import Base
    engine = _get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("all_postgres_tables_dropped")
