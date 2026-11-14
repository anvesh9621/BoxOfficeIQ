"""BoxOfficeIQ Database Configuration and Session Management.
Supports PostgreSQL (production) and SQLite (local development/testing).
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Generator, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from app.config import settings

logger = logging.getLogger(__name__)


# Platform-independent GUID/UUID type
class GUID(TypeDecorator):
    """Platform-independent GUID/UUID type.
    Uses PostgreSQL's UUID type when on Postgres, otherwise CHAR(36).
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return value
        if dialect.name == "postgresql":
            return str(value) if isinstance(value, uuid.UUID) else value
        else:
            return str(value) if isinstance(value, uuid.UUID) else str(uuid.UUID(str(value)))

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            try:
                return uuid.UUID(str(value))
            except (ValueError, TypeError):
                return value
        return value


# Base class for all ORM models
Base = declarative_base()


# Normalizing database URL for standard drivers
db_url = settings.database_url
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

connect_args = {}
if db_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# Primary sync engine (using psycopg2 or sqlite3)
engine = create_engine(
    db_url,
    echo=settings.debug,
    pool_pre_ping=True,
    connect_args=connect_args,
)

# Sync Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database session with automatic cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables registered with Base metadata."""
    # Ensure all models are imported before creating tables
    import app.models  # noqa: F401

    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized successfully.")


# Async compatibility layer (if async drivers are available)
try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

    async_db_url = db_url
    if async_db_url.startswith("postgresql://"):
        async_db_url = async_db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif async_db_url.startswith("sqlite:///"):
        async_db_url = async_db_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)

    async_engine = create_async_engine(
        async_db_url,
        echo=settings.debug,
        pool_pre_ping=True,
    )
    AsyncSessionLocal = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def get_async_db():
        async with AsyncSessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()

except (ImportError, Exception):
    # Async driver not installed or supported in current environment;
    # Fallback placeholders are provided so imports succeed.
    async_engine = None
    AsyncSessionLocal = None
    get_async_db = None
