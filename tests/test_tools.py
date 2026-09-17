from datetime import datetime, timedelta, timezone

from app.models import Booking
from app.tools import ToolPermissionError, check_fare_rule, execute_tool


def test_check_fare_rule_finds_seeded_booking(db_session):
    result = check_fare_rule(db_session, "WL4B92")
    assert result["found"] is True
    assert result["fare_class"] == "basic"
    assert result["refundable"] is False
    # This is the regression case for the SQLite naive/aware datetime bug:
    # subtracting an aware "now" from a value SQLite returned as naive used to
    # raise TypeError instead of returning a result.
    assert isinstance(result["days_until_travel"], int)


def test_check_fare_rule_handles_naive_datetime_from_sqlite(db_session):
    """Directly reproduce the bug: a booking whose travel_date comes back naive.

    SQLite silently drops timezone info even on a DateTime(timezone=True) column,
    so this is not a hypothetical — it's what every read from the real database
    looks like.
    """
    naive_future = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=10)
    db_session.add(
        Booking(
            booking_ref="NAIVE1",
            client_name="Test Traveler",
            fare_class="standard",
            refundable=True,
            change_fee=100.0,
            travel_date=naive_future,
        )
    )
    db_session.commit()

    result = check_fare_rule(db_session, "naive1")  # also exercises case-insensitivity
    assert result["found"] is True
    assert 8 <= result["days_until_travel"] <= 10


def test_check_fare_rule_missing_booking_returns_not_found(db_session):
    result = check_fare_rule(db_session, "NOSUCHBOOKING")
    assert result == {"booking_ref": "NOSUCHBOOKING", "found": False}


def test_execute_tool_enforces_role_permissions(db_session):
    # Both defined roles may use it — this just confirms the permission table
    # actually gates on the role argument rather than always allowing.
    execute_tool(db_session, "check_fare_rule", {"booking_ref": "WL4B92"}, role="agent")
    execute_tool(db_session, "check_fare_rule", {"booking_ref": "WL4B92"}, role="manager")
    try:
        execute_tool(db_session, "check_fare_rule", {"booking_ref": "WL4B92"}, role="nobody")
        assert False, "expected ToolPermissionError"
    except ToolPermissionError:
        pass
