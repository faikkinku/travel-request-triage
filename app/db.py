"""Database connection and session handling.

One engine for the whole process, and one short-lived session per request. The
session is what actually talks to the database; opening one per request keeps
requests from stepping on each other's transactions.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import DATABASE_URL

# check_same_thread is a SQLite-only quirk: it otherwise refuses to be used from
# the worker threads FastAPI runs sync endpoints on.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Parent class of every table. SQLAlchemy collects the schema through it."""


def get_db() -> Iterator[Session]:
    """FastAPI dependency: hand a session to an endpoint, close it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all() -> None:
    """Create every table that doesn't exist yet.

    No Alembic in this project (scope cut, documented in the README) — the schema is
    small and stable enough that create-if-missing is sufficient for a portfolio demo.
    Safe to call on every startup: existing tables are left untouched.
    """
    from app import models  # noqa: F401  (import registers the tables on Base)

    Base.metadata.create_all(engine)
