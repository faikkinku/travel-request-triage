"""Logins and roles.

Two roles: agent, manager. Checks live here and are called by the API layer, so a
permission decision happens on the server on every request. Hiding a button in a
template is not a permission check — anyone can type the URL.

Passwords are stored as a PBKDF2 hash with a per-user salt, using only the standard
library. A real agency would use a maintained password library; this is
deliberately small enough to read in one sitting.
"""

import hashlib
import hmac
import os

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User

MANAGER_ROLES = {"manager"}
_ITERATIONS = 260_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations)
        )
    except (ValueError, TypeError):
        return False
    # Constant-time comparison: a plain == leaks how much of the hash matched.
    return hmac.compare_digest(digest.hex(), digest_hex)


def authenticate(db: Session, username: str, password: str) -> User | None:
    user = db.scalar(select(User).where(User.username == username))
    if user and verify_password(password, user.password_hash):
        return user
    return None


def current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    """The logged-in user, or None. Never raises — pages may be public."""
    user_id = request.session.get("user_id")
    if user_id is None:
        return None
    return db.get(User, user_id)


def require_manager(user: User | None = Depends(current_user)) -> User:
    """Dependency for anything only a manager may reach.

    Applied to the endpoint itself, so typing the URL directly fails exactly the
    same way as clicking a hidden link would.
    """
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "sign in required")
    if user.role not in MANAGER_ROLES:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "manager only")
    return user
