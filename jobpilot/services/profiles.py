"""User profiles: resume parsing (structured output), preferences (long-term memory)."""
from __future__ import annotations

import json
import logging
from typing import Any

from .. import db
from ..llm import ChatLLM, Embedder, prompts
from ..schemas import CandidateProfile
from ..schemas.api import PreferencesIn
from ..utils import extract_skills, normalize_skills

log = logging.getLogger(__name__)


def parse_resume(llm: ChatLLM, resume_text: str) -> tuple[CandidateProfile, dict]:
    """LLM structured output; falls back to keyword extraction so signup never fails."""
    try:
        profile, stats = llm.structured(
            [
                {"role": "system", "content": prompts.RESUME_PARSER},
                {"role": "user", "content": resume_text[:12000]},
            ],
            CandidateProfile,
        )
        profile.skills = normalize_skills(profile.skills)[:25]
        return profile, {**stats, "source": "llm"}
    except Exception as exc:  # provider down, quota exhausted, invalid output...
        log.warning("resume parsing via LLM failed (%s); using keyword fallback", exc)
        lower = resume_text.lower()
        seniority = "intern" if ("b.tech" in lower or "student" in lower or "pursuing" in lower) else "fresher"
        return CandidateProfile(
            headline=resume_text.strip().splitlines()[0][:120] if resume_text.strip() else "Candidate",
            skills=extract_skills(resume_text)[:25],
            target_roles=[],
            years_experience=0,
            seniority=seniority,
            locations=[],
            education="",
        ), {"source": "fallback", "error": type(exc).__name__}


def effective_profile(profile: dict, prefs: dict) -> dict:
    """Merge the parsed resume with preferences the user told the agent."""
    skills = normalize_skills((profile.get("skills") or []) + (prefs.get("extra_skills") or []))
    return {
        "headline": profile.get("headline", ""),
        "skills": skills,
        "target_roles": prefs.get("target_roles") or profile.get("target_roles") or [],
        "years_experience": profile.get("years_experience"),
        "seniority": profile.get("seniority"),
        "locations": prefs.get("locations") or profile.get("locations") or [],
        "remote_ok": prefs.get("remote_ok", True),
        "min_salary_lpa": prefs.get("min_salary_lpa"),
    }


def profile_query(p: dict) -> str:
    """Text that represents what this person is looking for (used for embeddings + digest search)."""
    roles = ", ".join(p.get("target_roles") or []) or (p.get("headline") or "software engineer")
    return f"{roles}. Skills: {', '.join((p.get('skills') or [])[:15])}. Level: {p.get('seniority') or 'any'}."


def _store_embedding(c, user_id: str, eff: dict, embedder: Embedder) -> None:
    try:
        vec = embedder.embed([profile_query(eff)], kind="query")[0]
    except Exception:  # embedding API down: keep going, personalisation just gets weaker
        log.exception("could not embed profile for %s", user_id)
        return
    c.execute("UPDATE users SET profile_embedding = %s::vector WHERE id = %s", (db.to_vector(vec), user_id))


def create_user(llm: ChatLLM, embedder: Embedder, name: str, resume_text: str) -> dict:
    profile, stats = parse_resume(llm, resume_text)
    with db.conn() as c:
        row = c.execute(
            "INSERT INTO users (name, resume_text, profile) VALUES (%s, %s, %s) RETURNING id",
            (name, resume_text, db.jsonb(profile.model_dump())),
        ).fetchone()
        user_id = str(row["id"])
        _store_embedding(c, user_id, effective_profile(profile.model_dump(), {}), embedder)
    user = get_user(user_id)
    assert user is not None
    user["parse_stats"] = stats
    return user


def get_user(user_id: str) -> dict | None:
    try:
        with db.conn() as c:
            row = c.execute(
                "SELECT id, name, profile, prefs, profile_embedding::text AS emb, created_at FROM users WHERE id = %s",
                (user_id,),
            ).fetchone()
    except Exception:  # invalid uuid etc.
        return None
    if not row:
        return None
    user: dict[str, Any] = dict(row)
    user["id"] = str(user["id"])
    emb = user.pop("emb")
    user["profile_embedding_list"] = json.loads(emb) if emb else None
    user["effective_profile"] = effective_profile(user["profile"] or {}, user["prefs"] or {})
    return user


def update_preferences(embedder: Embedder, user_id: str, prefs_in: PreferencesIn) -> dict:
    user = get_user(user_id)
    if user is None:
        raise KeyError(user_id)
    prefs = dict(user["prefs"] or {})
    for key, value in prefs_in.model_dump(exclude_none=True).items():
        if key == "extra_skills":  # skills accumulate
            prefs[key] = normalize_skills([*(prefs.get(key) or []), *value])
        else:  # roles, cities, remote, salary: the latest statement wins
            prefs[key] = value
    with db.conn() as c:
        c.execute("UPDATE users SET prefs = %s WHERE id = %s", (db.jsonb(prefs), user_id))
        _store_embedding(c, user_id, effective_profile(user["profile"] or {}, prefs), embedder)
    return get_user(user_id)  # type: ignore[return-value]


def public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "name": user.get("name", ""),
        "profile": user.get("effective_profile"),
        "parse_stats": user.get("parse_stats"),
    }
