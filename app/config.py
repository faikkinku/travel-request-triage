"""Settings, read once from the environment.

Secrets never live in code. Locally they come from a git-ignored `.env`; in
production they come from the host's environment-variable settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

AGENCY_NAME = os.getenv("AGENCY_NAME", "Wanderlux Travel")


def normalise_database_url(raw: str) -> str:
    """Turn whatever the host handed us into a URL SQLAlchemy can actually open.

    Several hosts still give out the old `postgres://` scheme, which SQLAlchemy
    rejects, and a bare `postgresql://` resolves to psycopg2, which this project does
    not install — requirements.txt pins psycopg 3 — so the URL has to name that driver
    or the first connection dies with `ModuleNotFoundError: No module named 'psycopg2'`.
    """
    if raw.startswith("postgres://"):
        raw = raw.replace("postgres://", "postgresql://", 1)
    if raw.startswith("postgresql://"):
        raw = raw.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw


DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    (PROJECT_ROOT / "data").mkdir(exist_ok=True)
    DATABASE_URL = f"sqlite:///{PROJECT_ROOT / 'data' / 'triage.db'}"
else:
    DATABASE_URL = normalise_database_url(DATABASE_URL)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "8000"))

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

# Signs the session cookie. A fixed development value is fine locally; production
# must set its own, or every restart would log everyone out.
SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-not-for-production")

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


def agent_is_configured() -> bool:
    """Whether we can actually call the model, or must degrade gracefully."""
    return bool(ANTHROPIC_API_KEY)
