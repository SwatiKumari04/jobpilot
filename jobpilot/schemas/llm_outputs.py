"""Pydantic models the LLM must fill in (structured outputs)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Seniority = Literal["intern", "fresher", "junior", "mid", "senior"]


class CandidateProfile(BaseModel):
    """What the LLM extracts from a pasted resume."""
    headline: str = Field(description="One line, e.g. 'Final-year EE student, Python backend + ML'")
    skills: list[str] = Field(description="Technical skills, lowercase, most important first, max 25")
    target_roles: list[str] = Field(description="2-4 job titles this person should apply for")
    years_experience: float = Field(description="Full-time-equivalent years of work experience; internships count 0.25 each")
    seniority: Seniority
    locations: list[str] = Field(description="Cities the person mentions or lives in; empty if unknown")
    education: str = Field(description="Highest degree and college, short")


class DigestPick(BaseModel):
    job_id: int = Field(description="Must be one of the candidate job ids given to you")
    fit_score: int = Field(ge=0, le=100, description="How well this job fits the candidate, 0-100")
    why: list[str] = Field(description="2-3 short, specific reasons grounded in the job and the profile")
    missing_skills: list[str] = Field(description="Skills the job asks for that the candidate lacks (may be empty)")


class Digest(BaseModel):
    """Elip-style morning digest: the 3 best jobs for this person today."""
    picks: list[DigestPick]
    summary: str = Field(description="One friendly sentence summarising today's picks")
