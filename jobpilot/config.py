"""All settings come from environment variables (or a local .env file).

Nothing secret is ever hard-coded. On Render you set these in the dashboard.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # reads .env when running locally; no-op in production

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str
    # Chat model (any OpenAI-compatible endpoint: Gemini, OpenAI, Groq, ...)
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    # Embedding model (can be a different provider from the chat model)
    embed_provider: str  # "openai" (any OpenAI-compatible API) or "hash" (offline, for tests)
    embed_api_key: str
    embed_base_url: str
    embed_model: str
    embed_dim: int
    admin_token: str
    rate_limit_per_10min: int
    log_level: str
    init_db_on_startup: bool


def get_settings() -> Settings:
    llm_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    return Settings(
        database_url=os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/jobpilot"),
        llm_api_key=llm_key,
        llm_base_url=os.getenv("LLM_BASE_URL", GEMINI_OPENAI_BASE_URL),
        llm_model=os.getenv("LLM_MODEL", "gemini-3.5-flash-lite"),
        embed_provider=os.getenv("EMBED_PROVIDER", "openai"),
        embed_api_key=os.getenv("EMBED_API_KEY") or llm_key,
        embed_base_url=os.getenv("EMBED_BASE_URL", os.getenv("LLM_BASE_URL", GEMINI_OPENAI_BASE_URL)),
        embed_model=os.getenv("EMBED_MODEL", "gemini-embedding-2"),
        embed_dim=int(os.getenv("EMBED_DIM", "768")),
        admin_token=os.getenv("ADMIN_TOKEN", "change-me"),
        rate_limit_per_10min=int(os.getenv("RATE_LIMIT_PER_10MIN", "30")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        init_db_on_startup=_bool("INIT_DB_ON_STARTUP", True),
    )
