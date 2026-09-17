"""The database tables.

Eight tables, in three groups:

- The conversation:  agent_sessions -> messages -> proposals
- The record:        audit_log (append-only; every request writes here)
- The agency:         users, bookings, knowledge_chunks
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentSession(Base):
    """One travel agent's visit to the intake page."""

    __tablename__ = "agent_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    actor_label: Mapped[str] = mapped_column(String(120), default="anonymous")

    messages: Mapped[list["Message"]] = relationship(back_populates="session")
    proposals: Mapped[list["Proposal"]] = relationship(back_populates="session")


class Message(Base):
    """One thing that was said: the client's request, or the assistant's answer."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("agent_sessions.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    role: Mapped[str] = mapped_column(String(20))  # "client" | "assistant"
    content: Mapped[str] = mapped_column(Text)

    session: Mapped[AgentSession] = relationship(back_populates="messages")


class Proposal(Base):
    """A triage recommendation awaiting a human decision.

    The model's answer is stored whole in `result` rather than spread across columns:
    the schema in schemas.py is the contract, and this is that contract as recorded.
    Status moves pending -> approved / rejected, and only a manager can move it.
    """

    __tablename__ = "proposals"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("agent_sessions.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    result: Mapped[dict] = mapped_column(JSON)          # a serialized TriageResult
    override_flags: Mapped[list] = mapped_column(JSON, default=list)
    # Not a foreign key on purpose: an agent can type a booking reference that turns
    # out not to exist, and that is itself something worth recording, not hiding.
    booking_ref: Mapped[str | None] = mapped_column(String(40), nullable=True)
    # A serialized FareCheck — whatever check_fare_rule actually returned, including
    # a not-found result. Stored whole so the manager sees the same fare details the
    # agent saw, not just the reference string.
    fare_check: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    status: Mapped[str] = mapped_column(String(20), default="pending")
    decided_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    session: Mapped[AgentSession] = relationship(back_populates="proposals")


class AuditLog(Base):
    """Append-only record of everything that happened. Never updated, never deleted.

    Written before a request can fail, so a crash still leaves a trace. `details`
    holds the full prompt context, the full response, and token usage for model
    calls — the difference between "the agent said X" and being able to prove why.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    session_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actor: Mapped[str] = mapped_column(String(120), default="system")
    event: Mapped[str] = mapped_column(String(60))
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class User(Base):
    """Logins. Role decides what the API will let a user do."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(60), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20))  # agent | manager
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Booking(Base):
    """A stand-in for the agency's real reservation system.

    check_availability-style tools in these projects only earn their keep if they
    hit something real rather than letting the model guess — this table is that
    something, seeded with a handful of synthetic bookings in seed.py.
    """

    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(primary_key=True)
    booking_ref: Mapped[str] = mapped_column(String(40), unique=True)
    client_name: Mapped[str] = mapped_column(String(120))
    fare_class: Mapped[str] = mapped_column(String(20))  # basic | standard | flexible
    refundable: Mapped[bool] = mapped_column(Boolean, default=False)
    change_fee: Mapped[float] = mapped_column(Float, default=0.0)
    travel_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class KnowledgeChunk(Base):
    """A passage of an agency policy document.

    No stored vector: retrieval is TF-IDF, fit fresh over this small corpus (a few
    dozen chunks) at query time — see app/knowledge.py for why that is enough here,
    and the README for the upgrade path if the corpus grows.
    """

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document: Mapped[str] = mapped_column(String(120))
    section: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
