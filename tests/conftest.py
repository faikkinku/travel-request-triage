import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("SECRET_KEY", "test-secret")
# Force-disable the live model call for the whole test session, regardless of what
# the developer's shell has exported — these tests must never spend real API
# credits or depend on network access.
os.environ["ANTHROPIC_API_KEY"] = ""

from app import models  # noqa: F401  (import registers the tables on Base)
from app.auth import hash_password
from app.db import Base, get_db
from app.knowledge import index_knowledge
from app.main import app
from app.models import Booking, User

TEST_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(bind=TEST_ENGINE, autoflush=False, expire_on_commit=False)


def _override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture(scope="session", autouse=True)
def _seeded_database():
    Base.metadata.create_all(TEST_ENGINE)
    db = TestSessionLocal()
    db.add(User(username="agent1", password_hash=hash_password("agent123"), role="agent"))
    db.add(User(username="manager", password_hash=hash_password("manager123"), role="manager"))
    db.add(
        Booking(
            booking_ref="WL4B92",
            client_name="Jordan Reyes",
            fare_class="basic",
            refundable=False,
            change_fee=250.0,
            travel_date=datetime.now(timezone.utc) + timedelta(days=2),
        )
    )
    db.commit()
    index_knowledge(db)
    db.close()
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db_session():
    """A session bound to the same test engine the app itself uses in tests.

    Exposed as a fixture rather than an importable name: importing this module as
    `tests.conftest` from another test file creates a second module instance (since
    tests/ has no __init__.py), which would spin up a second in-memory engine with
    none of the tables or seed data the real one has.
    """
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()
