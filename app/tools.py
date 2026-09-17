"""Tools the model may call, and the rules about who may call them.

A tool is real code the model is allowed to run. We describe each one to the model;
it can then say "call check_fare_rule with booking_ref=ABC123", and this module runs
it and hands back the result — a real lookup against the bookings table, not a
number the model made up.

Permissions are enforced here rather than by omitting a tool from a prompt. A prompt
that leaves a tool out is a request; a function that refuses to execute it is a rule.
"""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Booking

# Which role may invoke which tool. Both roles in this app may look up a booking;
# neither the model nor either role can issue a refund or waive a fee directly —
# that stays a human action taken after reviewing the proposal.
ROLE_TOOLS: dict[str, set[str]] = {
    "agent": {"check_fare_rule"},
    "manager": {"check_fare_rule"},
}

TOOL_DEFINITIONS = [
    {
        "name": "check_fare_rule",
        "description": (
            "Look up a real booking by its reference to see the fare class, whether "
            "it is refundable, the change fee, and the travel date. Call this before "
            "proposing any fee waiver, refund, or rebooking action. Never guess a "
            "fare class or fee — always call this tool if the client's message "
            "mentions a booking reference."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "booking_ref": {
                    "type": "string",
                    "description": "The booking reference as given by the client, e.g. WL4B92.",
                }
            },
            "required": ["booking_ref"],
        },
    }
]


class ToolPermissionError(RuntimeError):
    """Raised when a role attempts a tool it is not allowed to use."""


def check_fare_rule(db: Session, booking_ref: str) -> dict:
    """Return what the reservation system knows about this booking, if anything."""
    booking = db.scalar(select(Booking).where(Booking.booking_ref == booking_ref.strip().upper()))
    if booking is None:
        return {"booking_ref": booking_ref, "found": False}

    # SQLite does not actually persist timezone offsets even on a
    # DateTime(timezone=True) column — values round-trip as naive. Every value we
    # write is UTC (see seed.py), so treat a naive read as UTC rather than let the
    # subtraction below raise on aware-minus-naive.
    travel_date = booking.travel_date
    if travel_date.tzinfo is None:
        travel_date = travel_date.replace(tzinfo=timezone.utc)
    days_until_travel = (travel_date - datetime.now(timezone.utc)).days
    return {
        "booking_ref": booking.booking_ref,
        "found": True,
        "fare_class": booking.fare_class,
        "refundable": booking.refundable,
        "change_fee": booking.change_fee,
        "travel_date": travel_date.isoformat(),
        "days_until_travel": days_until_travel,
    }


def execute_tool(db: Session, name: str, tool_input: dict, role: str = "agent") -> dict:
    """Run a tool on behalf of a role, refusing anything that role may not use."""
    allowed = ROLE_TOOLS.get(role, set())
    if name not in allowed:
        raise ToolPermissionError(f"role '{role}' may not use tool '{name}'")

    if name == "check_fare_rule":
        return check_fare_rule(db, tool_input.get("booking_ref", ""))

    raise ToolPermissionError(f"unknown tool '{name}'")
