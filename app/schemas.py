"""The output contract.

These classes define the exact shape of a triage result: which fields must exist,
what type each one is, and which values are allowed. `agent.py` hands this same
schema to the model, so whatever it produces is validated against these rules
before any other part of the app is allowed to see it.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class RequestType(str, Enum):
    """The kinds of client request this desk handles. Nothing outside this list is
    a valid answer."""

    REBOOKING = "rebooking"
    CANCELLATION = "cancellation"
    REFUND = "refund"
    DOCUMENT_ISSUE = "document_issue"
    COMPLAINT = "complaint"
    FARE_DISPUTE = "fare_dispute"
    GENERAL_INQUIRY = "general_inquiry"


class Urgency(str, Enum):
    """How soon a human needs to act on this request."""

    IMMEDIATE = "immediate"  # traveling within 24h, or already at the airport
    URGENT = "urgent"        # within 48 hours
    SOON = "soon"            # within a week
    ROUTINE = "routine"      # no travel-date pressure


class Citation(BaseModel):
    """One piece of evidence behind a claim.

    Filled from agency policy documents actually retrieved for this request, which
    is what makes an answer auditable: you can open the file and check the
    reasoning follows.
    """

    document: str = Field(description="Filename of the policy document, e.g. refund-and-cancellation-policy.md")
    section: str = Field(description="Heading or section within that document")
    quote: str = Field(description="The sentence from the document that supports the claim")


class IntakeRequest(BaseModel):
    """What a program sends us: the client's own words, nothing else."""

    message: str = Field(min_length=1, max_length=4000)


class TriageResult(BaseModel):
    """What we send back — the contract the model has to fill.

    `escalate` is deliberately separate from `urgency`. Urgency is a judgement about
    how soon the request needs attention; escalate means a manager must review it
    rather than a front-line agent acting alone. `guardrails.py` can force it to True
    regardless of what the model decided, which is what turns the safety promise
    into a mechanism rather than a request.
    """

    request_type: RequestType
    urgency: Urgency
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_reason: str = Field(
        description="Why this confidence — what is missing or uncertain. A bare number says nothing."
    )
    citations: list[Citation] = Field(
        description="The retrieved policy passages this result relies on. Never invent one."
    )
    escalate: bool
    escalate_reason: str | None = None
    proposed_action: str = Field(
        description="One sentence: what the agent should do next, grounded in policy and any fare-rule check"
    )
    summary: str = Field(description="One sentence restating the client's request in plain terms")


class FareCheck(BaseModel):
    """What check_fare_rule found for a booking reference, if it found anything."""

    booking_ref: str
    found: bool
    fare_class: str | None = None
    refundable: bool | None = None
    change_fee: float | None = None
    travel_date: datetime | None = None

    model_config = {"from_attributes": True}


class TriageResponse(BaseModel):
    """The full API response: the model's answer, plus what our own code decided.

    Keeping these apart matters. `result` is what the model said (already corrected
    by the independent guardrail check); `override_flags` is what plain Python found
    on its own. A caller can see both and never has to take the model's word for it.
    """

    proposal_id: int
    result: TriageResult
    override_flags: list[str]
    fare_check: FareCheck | None = None
    status: str
