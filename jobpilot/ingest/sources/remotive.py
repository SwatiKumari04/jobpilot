"""Fetch REAL remote tech jobs from Remotive's free public API and normalise them.

API docs: https://github.com/remotive-com/remote-jobs-api
Please respect their terms: link back to the Remotive job URL (the UI does)
and don't call it more than a few times a day (we run it nightly).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import httpx

from ...utils import extract_skills, strip_html
from ..pipeline import JobIn

log = logging.getLogger(__name__)
API = "https://remotive.com/api/remote-jobs"
DEFAULT_CATEGORIES = ["software-dev", "data", "devops", "qa"]

_SENIOR = re.compile(r"\b(senior|sr\.?|lead|staff|principal|head|director|manager|architect)\b", re.I)
_JUNIOR = re.compile(r"\b(junior|jr\.?|entry|graduate|new grad|trainee|associate)\b", re.I)
_INTERN = re.compile(r"\bintern(ship)?\b", re.I)
_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:-|to)?\s*\d{0,2}\s*\+?\s*years?", re.I)
_CATEGORY_MAP = {"Software Development": "software", "Data Analysis": "data", "Data": "data",
                 "DevOps / Sysadmin": "devops", "QA": "qa"}


def infer_level(title: str, description: str) -> tuple[str, float, float]:
    """Guess level + experience range from the title and '3+ years' style phrases."""
    years = [int(m.group(1)) for m in _YEARS.finditer(description[:4000]) if int(m.group(1)) <= 20]
    min_years = min(years) if years else None
    if _INTERN.search(title):
        return "intern", 0, 0.5
    if _SENIOR.search(title):
        return "senior", float(min_years if min_years is not None else 5), 15
    if _JUNIOR.search(title):
        return "junior", float(min_years if min_years is not None else 0), 3
    if min_years is not None:
        return ("junior" if min_years <= 2 else "mid"), float(min_years), float(min_years + 5)
    return "mid", 2, 8


def parse_date(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def normalise(raw: dict) -> JobIn | None:
    try:
        title = (raw.get("title") or "").strip()
        company = (raw.get("company_name") or "").strip()
        if not title or not company:
            return None
        text = strip_html(raw.get("description") or "")
        salary = (raw.get("salary") or "").strip()
        if salary:
            text = f"Salary: {salary}\n{text}"
        level, emin, emax = infer_level(title, text)
        tags = [t for t in (raw.get("tags") or []) if isinstance(t, str)]
        skills = list(dict.fromkeys(extract_skills(", ".join(tags)) + extract_skills(f"{title}\n{text}")))[:15]
        return JobIn(
            source="remotive",
            external_id=str(raw["id"]),
            title=title,
            company=company,
            location=(raw.get("candidate_required_location") or "Worldwide").strip(),
            remote=True,
            category=_CATEGORY_MAP.get(raw.get("category") or "", (raw.get("category") or "").lower()),
            level=level,
            experience_min=emin,
            experience_max=emax,
            skills=skills,
            description=text[:8000],
            url=raw.get("url") or "",
            posted_at=parse_date(raw.get("publication_date")),
        )
    except Exception:
        log.exception("could not normalise remotive job %s", raw.get("id"))
        return None


def fetch(categories: list[str] | None = None, limit_per_category: int = 150, timeout: float = 30) -> list[JobIn]:
    jobs: dict[str, JobIn] = {}
    with httpx.Client(timeout=timeout, headers={"User-Agent": "JobPilot student project"}) as client:
        for cat in categories or DEFAULT_CATEGORIES:
            resp = client.get(API, params={"category": cat, "limit": limit_per_category})
            resp.raise_for_status()
            raw_jobs = resp.json().get("jobs", [])
            log.info("remotive %s: %d jobs", cat, len(raw_jobs))
            for raw in raw_jobs:
                job = normalise(raw)
                if job:
                    jobs[job.external_id] = job
    return list(jobs.values())
