"""Application configuration – loads settings from .env file."""

from datetime import datetime, timezone
import os
from typing import Any
from unittest.mock import NonCallableMock
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

DEFAULT_GEMINI_MODEL: str = "gemini-3.5-flash"
DEFAULT_GEMINI_FALLBACK_MODEL: str = "gemini-2.5-flash"
_DEFAULT_CORS_ORIGINS: str = (
    "http://localhost:3000,http://localhost:8000,"
    "http://127.0.0.1:3000,http://127.0.0.1:8000"
)

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", DEFAULT_GEMINI_MODEL)
GEMINI_FALLBACK_MODEL_NAME: str = os.getenv(
    "GEMINI_FALLBACK_MODEL_NAME", DEFAULT_GEMINI_FALLBACK_MODEL
)
GOOGLE_CREDENTIALS_PATH: str = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
GOOGLE_TOKEN_PATH: str = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
TIMEZONE: str = os.getenv("TIMEZONE", "Africa/Cairo")
CORS_ALLOW_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]


def get_today_date_str() -> str:
    """Return today's calendar date (YYYY-MM-DD) in the configured timezone."""
    tz_name = TIMEZONE or "Africa/Cairo"
    try:
        tz: ZoneInfo | timezone = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    return datetime.now(tz).strftime("%Y-%m-%d")


def get_gemini_candidate_models() -> list[str]:
    """Return ordered, deduplicated primary and fallback Gemini model names."""
    primary = GEMINI_MODEL_NAME or DEFAULT_GEMINI_MODEL
    fallback = GEMINI_FALLBACK_MODEL_NAME or DEFAULT_GEMINI_FALLBACK_MODEL
    candidates: list[str] = [primary]
    if fallback and fallback not in candidates:
        candidates.append(fallback)
    elif DEFAULT_GEMINI_MODEL not in candidates:
        candidates.append(DEFAULT_GEMINI_MODEL)
    elif "gemini-2.0-flash" not in candidates:
        candidates.append("gemini-2.0-flash")
    return candidates


def is_mocked_llm_class(llm_cls: Any) -> bool:
    """Return True when *llm_cls* has been replaced by a test fake or mock."""
    if isinstance(llm_cls, NonCallableMock):
        return True
    mod = getattr(llm_cls, "__module__", "") or ""
    qual = getattr(llm_cls, "__qualname__", "") or ""
    return not (mod == "langchain_google_genai.chat_models" and qual == "ChatGoogleGenerativeAI")


def can_invoke_llm(llm_cls: Any) -> bool:
    """Return True when invoking *llm_cls* is allowed and configured.

    Never opens real external network sockets under pytest unless *llm_cls*
    has been mocked/faked by the test suite.
    """
    if not GEMINI_API_KEY:
        return False
    if is_mocked_llm_class(llm_cls):
        return True
    return not bool(os.getenv("PYTEST_CURRENT_TEST"))
