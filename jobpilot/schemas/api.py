"""Request bodies for the HTTP API (validated by FastAPI)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..ingest.pipeline import JobIn


class CreateUserIn(BaseModel):
    name: str = Field(default="", max_length=100)
    resume_text: str = Field(min_length=30, max_length=20000)


class PreferencesIn(BaseModel):
    target_roles: list[str] | None = None
    locations: list[str] | None = None
    remote_ok: bool | None = None
    min_salary_lpa: float | None = Field(default=None, ge=0, le=500)
    extra_skills: list[str] | None = None


class ChatIn(BaseModel):
    user_id: str
    message: str = Field(min_length=1, max_length=2000)


class FeedbackIn(BaseModel):
    user_id: str
    job_id: int
    signal: Literal[1, -1]


class IngestIn(BaseModel):
    jobs: list[JobIn]
    full_snapshot_of_source: str | None = None
