"""Fill an empty database with a working agency, and index the knowledge base.

Run:

    python seed.py

Idempotent: existing users and bookings are left alone; the knowledge index is
always rebuilt from the files in knowledge/, because those are the source of truth.
Also creates the tables if they don't exist yet (see app/db.py — no Alembic here).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.auth import hash_password
from app.db import SessionLocal, create_all
from app.knowledge import index_knowledge
from app.models import Booking, User

DEMO_USERS = [
    ("agent1", "agent123", "agent"),
    ("agent2", "agent123", "agent"),
    ("manager", "manager123", "manager"),
]

# A handful of synthetic bookings so check_fare_rule has something real to find.
# travel_offset_days is relative to when seed.py runs, so the demo always has a mix
# of imminent, near-term, and far-out travel dates.
BOOKINGS = [
    ("WL4B92", "Jordan Reyes", "basic", False, 250.0, 2),      # travels in 2 days
    ("WL7K10", "Priya Nair", "standard", True, 100.0, 5),
    ("WL2X77", "Sam O'Connor", "flexible", True, 0.0, 14),
    ("WL9Q41", "Maria Lindqvist", "basic", False, 250.0, 30),
    ("WL5T63", "Daniel Osei", "standard", True, 100.0, 60),
    ("WL1M08", "Hana Kobayashi", "flexible", True, 0.0, 1),    # travels tomorrow
]


def seed_users(db) -> int:
    created = 0
    for username, password, role in DEMO_USERS:
        if db.scalar(select(User).where(User.username == username)):
            continue
        db.add(User(username=username, password_hash=hash_password(password), role=role))
        created += 1
    db.commit()
    return created


def seed_bookings(db) -> int:
    created = 0
    now = datetime.now(timezone.utc)
    for ref, name, fare_class, refundable, fee, offset_days in BOOKINGS:
        if db.scalar(select(Booking).where(Booking.booking_ref == ref)):
            continue
        db.add(
            Booking(
                booking_ref=ref,
                client_name=name,
                fare_class=fare_class,
                refundable=refundable,
                change_fee=fee,
                travel_date=now + timedelta(days=offset_days),
            )
        )
        created += 1
    db.commit()
    return created


def main() -> None:
    create_all()
    with SessionLocal() as db:
        print(f"users created:    {seed_users(db)}")
        print(f"bookings created: {seed_bookings(db)}")
        print(f"chunks indexed:   {index_knowledge(db)}")

    print("\nSign-in details for the demo:")
    for username, password, role in DEMO_USERS:
        print(f"  {role:<8} {username} / {password}")
    print("\nSample booking references to try: " + ", ".join(b[0] for b in BOOKINGS))


if __name__ == "__main__":
    main()
