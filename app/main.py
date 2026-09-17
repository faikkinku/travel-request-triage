"""The web application: HTML pages for humans, JSON endpoints for programs.

Both doors run the same code path (`process_intake`), so the browser and the API
can never drift apart in what they decide — only in how they present it.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from app import config
from app.agent import AgentError, AgentNotConfigured, run_agent
from app.auth import authenticate, current_user, require_manager
from app.db import create_all, get_db
from app.guardrails import apply_guardrails, find_injection_attempts, find_overrides
from app.models import AgentSession, AuditLog, Message, Proposal, User, utcnow
from app.schemas import IntakeRequest, TriageResponse, TriageResult

logger = logging.getLogger(__name__)

BASE_DIR = config.BASE_DIR

@asynccontextmanager
async def lifespan(_app: FastAPI):
    create_all()
    # Seed on every startup, not just once at deploy time. Render's free tier has
    # no persistent disk: the SQLite fallback file is wiped every time the service
    # spins back up after an idle period, so a one-time pre-deploy seed would leave
    # the app with no login users after the first spin-down. seed_users/
    # seed_bookings/index_knowledge are all idempotent, so re-running them against
    # an already-seeded database (Postgres in production, where data persists) is
    # a fast no-op rather than a problem.
    from app.db import SessionLocal
    from app.knowledge import index_knowledge
    from seed import seed_bookings, seed_users

    with SessionLocal() as db:
        seed_users(db)
        seed_bookings(db)
        index_knowledge(db)
    yield


app = FastAPI(title="Travel Request Triage", version="1.0.0", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=config.SECRET_KEY, https_only=False)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def audit(db: Session, event: str, *, actor: str = "system", session_id: int | None = None, **details) -> None:
    """Append one row to the audit log and commit it immediately.

    Committed on its own so the record survives a failure later in the request. An
    audit log that is only written when everything succeeds records the wrong half
    of history.
    """
    db.add(AuditLog(event=event, actor=actor, session_id=session_id, details=details))
    db.commit()


# --------------------------------------------------------------------------- intake


def _guardrail_fields(result: TriageResult) -> dict:
    """The only three fields apply_guardrails can change — see guardrails.py.

    Recording exactly these keeps the before/after pair honest: an override that
    rewrote nothing but the reason used to show two identical rows.
    """
    return {
        "urgency": result.urgency.value,
        "escalate": result.escalate,
        "escalate_reason": result.escalate_reason,
    }


def process_intake(db: Session, message: str, actor: str = "anonymous", role: str = "agent") -> TriageResponse:
    """Run one client request all the way through: store, reason, check, record.

    Order matters. The message is stored and audited before the model is called,
    the model's answer is corrected by guardrails.py before anyone sees it, and the
    proposal is saved as *pending* — never as an action already taken.
    """
    agent_session = AgentSession(actor_label=actor)
    db.add(agent_session)
    db.flush()
    db.add(Message(session_id=agent_session.id, role="client", content=message))
    db.commit()

    audit(
        db,
        "intake_received",
        actor=actor,
        session_id=agent_session.id,
        message_chars=len(message),
        overrides_precheck=find_overrides(message),
        injection_precheck=find_injection_attempts(message),
    )

    raw_result, _chunks, fare_check, agent_audit = run_agent(db, message, role=role)
    audit(
        db,
        "model_call",
        actor=actor,
        session_id=agent_session.id,
        **agent_audit,
    )

    result, override_flags = apply_guardrails(raw_result, message)
    if result.model_dump() != raw_result.model_dump():
        audit(
            db,
            "guardrail_override",
            actor="guardrail_check",
            session_id=agent_session.id,
            before=_guardrail_fields(raw_result),
            after=_guardrail_fields(result),
            override_flags=override_flags,
        )

    proposal = Proposal(
        session_id=agent_session.id,
        result=json.loads(result.model_dump_json()),
        override_flags=override_flags,
        booking_ref=fare_check.booking_ref if fare_check else None,
        fare_check=json.loads(fare_check.model_dump_json()) if fare_check else None,
        status="pending",
    )
    db.add(proposal)
    db.add(
        Message(
            session_id=agent_session.id,
            role="assistant",
            content=result.model_dump_json(),
        )
    )
    db.commit()

    audit(
        db,
        "proposal_created",
        actor=actor,
        session_id=agent_session.id,
        proposal_id=proposal.id,
        request_type=result.request_type.value,
        urgency=result.urgency.value,
        escalate=result.escalate,
        citations=[c.document for c in result.citations],
    )

    return TriageResponse(
        proposal_id=proposal.id,
        result=result,
        override_flags=override_flags,
        fare_check=fare_check,
        status=proposal.status,
    )


# ------------------------------------------------------------------------ HTML pages


@app.get("/", response_class=HTMLResponse)
def intake_page(request: Request, user: User | None = Depends(current_user)):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "agency_name": config.AGENCY_NAME,
            "user": user,
            "agent_ready": config.agent_is_configured(),
        },
    )


@app.post("/", response_class=HTMLResponse)
def submit_intake(
    request: Request,
    message: str = Form(...),
    db: Session = Depends(get_db),
    user: User | None = Depends(current_user),
):
    context: dict = {
        "agency_name": config.AGENCY_NAME,
        "user": user,
        "agent_ready": config.agent_is_configured(),
        "message": message,
    }
    try:
        response = process_intake(
            db,
            message,
            actor=user.username if user else "anonymous",
            role=user.role if user else "agent",
        )
        context["response"] = response
    except AgentNotConfigured as exc:
        # Still show what the independent check found — that part needs no model,
        # and it is the half an agency would actually care about first.
        context["error"] = str(exc)
        context["overrides"] = find_overrides(message)
    except AgentError as exc:
        context["error"] = f"The assistant could not complete this triage: {exc}"
        context["overrides"] = find_overrides(message)

    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="login.html", context={"agency_name": config.AGENCY_NAME}
    )


@app.post("/login")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = authenticate(db, username, password)
    if user is None:
        audit(db, "login_failed", actor=username)
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"agency_name": config.AGENCY_NAME, "error": "Wrong username or password."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    request.session["user_id"] = user.id
    audit(db, "login", actor=user.username, role=user.role)
    destination = "/review" if user.role == "manager" else "/"
    return RedirectResponse(destination, status_code=status.HTTP_303_SEE_OTHER)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/review", response_class=HTMLResponse)
def review_queue(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_manager),
):
    """The manager queue. The dependency above is the access control — not the template."""
    proposals = list(
        db.scalars(select(Proposal).order_by(Proposal.created_at.desc()).limit(50))
    )
    rows = []
    for proposal in proposals:
        try:
            result = TriageResult.model_validate(proposal.result)
        except ValidationError:
            result = None
        rows.append({"proposal": proposal, "result": result})
    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={"agency_name": config.AGENCY_NAME, "user": user, "rows": rows},
    )


@app.post("/review/{proposal_id}/decide")
def decide(
    proposal_id: int,
    decision: str = Form(...),
    note: str = Form(""),
    db: Session = Depends(get_db),
    user: User = Depends(require_manager),
):
    if decision not in {"approve", "reject"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "decision must be approve or reject")

    proposal = db.get(Proposal, proposal_id)
    if proposal is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such proposal")
    if proposal.status != "pending":
        raise HTTPException(status.HTTP_409_CONFLICT, f"already {proposal.status}")

    proposal.status = "approved" if decision == "approve" else "rejected"
    proposal.decided_by = user.username
    proposal.decided_at = utcnow()
    proposal.decision_note = note or None
    db.commit()

    audit(
        db,
        "proposal_decided",
        actor=user.username,
        session_id=proposal.session_id,
        proposal_id=proposal.id,
        decision=proposal.status,
        note=note or None,
    )
    return RedirectResponse("/review", status_code=status.HTTP_303_SEE_OTHER)


# --------------------------------------------------------------------- JSON endpoints


@app.post("/api/intake")
def api_intake(
    intake: IntakeRequest,
    db: Session = Depends(get_db),
    user: User | None = Depends(current_user),
) -> TriageResponse:
    """The machine-facing door: JSON in, a validated TriageResponse out."""
    try:
        return process_intake(
            db,
            intake.message,
            actor=user.username if user else "api-anonymous",
            role=user.role if user else "agent",
        )
    except AgentNotConfigured as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except AgentError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc


@app.post("/api/guardrail-check")
def api_guardrail_check(intake: IntakeRequest) -> dict:
    """Run only the independent override check. No model, no database, no key needed.

    Exposed on its own because it is the one part of the system whose answer does
    not depend on anything unpredictable — useful for testing and for proving the
    check runs regardless of what the model would have said.
    """
    return {
        "overrides": find_overrides(intake.message),
        "injection_attempts": find_injection_attempts(intake.message),
    }


@app.get("/api/proposals")
def api_proposals(
    db: Session = Depends(get_db),
    _manager: User = Depends(require_manager),
) -> list[dict]:
    """Manager-only. Same rule as the HTML queue, enforced in the same place."""
    proposals = db.scalars(select(Proposal).order_by(Proposal.created_at.desc()).limit(50))
    return [
        {
            "id": p.id,
            "created_at": p.created_at.isoformat(),
            "status": p.status,
            "override_flags": p.override_flags,
            "booking_ref": p.booking_ref,
            "result": p.result,
        }
        for p in proposals
    ]


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "agent_configured": config.agent_is_configured()}


@app.post("/api/intake/broken")
def api_intake_broken() -> JSONResponse:
    """Deliberately build an invalid result, to show the contract rejecting it.

    Kept as a demonstration: the errors returned here are exactly what would be
    caught if the model returned nonsense, which is why nonsense never reaches the
    database.
    """
    try:
        TriageResult(
            request_type="lost luggage",  # not one of the allowed request types
            urgency="pretty bad",         # not one of the allowed urgency levels
            confidence=1.7,               # outside 0.0-1.0
            confidence_reason="",
            citations=[],
            escalate="maybe",             # not a boolean
            proposed_action="",
            summary="",
        )
    except ValidationError as exc:
        return JSONResponse(status_code=422, content=json.loads(exc.json()))
    return JSONResponse(status_code=500, content={"error": "validation should have failed"})
