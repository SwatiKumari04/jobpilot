"""Ranking and personalisation — pure functions, fully unit-tested.

Pipeline:  retrieve (keyword + vector)  ->  fuse with RRF  ->  personalised re-rank.
Every score comes with human-readable reasons, so the UI (and the LLM) can
explain *why* a job was recommended.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable

RRF_K = 60  # standard constant from the Reciprocal Rank Fusion paper (Cormack et al., 2009)


def rrf_fuse(rank_lists: Iterable[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank(d)).

    Works on ranks only, so it can merge BM25-style keyword scores and cosine
    similarities without having to calibrate them against each other.
    """
    scores: dict[int, float] = {}
    for ranked in rank_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def skill_overlap(job_skills: list[str], user_skills: list[str]) -> tuple[float, list[str], list[str]]:
    """Fraction of the job's skills the user has (capped at 6 so long lists aren't punished)."""
    js = [s.lower() for s in job_skills]
    us = {s.lower() for s in user_skills}
    if not js:
        return 0.5, [], []
    matched = [s for s in js if s in us]
    missing = [s for s in js if s not in us]
    return min(1.0, len(matched) / min(len(js), 6)), matched, missing


def location_fit(job_location: str, job_remote: bool, user_locations: list[str], remote_ok: bool | None) -> tuple[float, str | None]:
    if job_remote and remote_ok is not False:
        return 1.0, "Remote"
    loc = (job_location or "").lower()
    for u in user_locations:
        if u and u.lower() in loc:
            return 1.0, f"In {u.title()}"
    if not user_locations:
        return 0.6, None
    return 0.2, None


def experience_fit(user_years: float | None, exp_min: float, exp_max: float) -> tuple[float, str | None]:
    if user_years is None:
        return 0.6, None
    if exp_min <= user_years + 0.5 and user_years <= exp_max + 2:
        label = f"{exp_min:g}-{exp_max:g} yrs" if exp_max < 30 else f"{exp_min:g}+ yrs"
        return 1.0, f"Fits your experience ({label})"
    if user_years < exp_min:
        return max(0.0, 1.0 - (exp_min - user_years) / 3.0), None
    return 0.7, None


def recency(posted_at: datetime, now: datetime | None = None) -> tuple[float, int]:
    now = now or datetime.now(timezone.utc)
    age_days = max(0, int((now - posted_at).total_seconds() // 86400))
    return math.exp(-age_days / 30.0), age_days


def score_job(job: dict, profile: dict | None, rel: float, now: datetime | None = None) -> tuple[float, list[str], list[str]]:
    """Combine relevance (0..1, from RRF) with personal signals.

    job: row from the jobs table plus optional p_sim / like_sim / dislike_sim.
    profile: {"skills", "years_experience", "locations", "remote_ok", "min_salary_lpa"} or None.
    Returns (score, reasons, missing_skills).
    """
    rec, age_days = recency(job["posted_at"], now)
    ghost = float(job.get("ghost_score") or 0.0)
    reasons: list[str] = []
    missing: list[str] = []

    if not profile:
        score = rel + 0.05 * rec - 0.3 * ghost
    else:
        so, matched, missing = skill_overlap(job.get("skills") or [], profile.get("skills") or [])
        lf, loc_reason = location_fit(job.get("location", ""), bool(job.get("remote")),
                                      profile.get("locations") or [], profile.get("remote_ok"))
        ef, exp_reason = experience_fit(profile.get("years_experience"),
                                        float(job.get("experience_min") or 0), float(job.get("experience_max") or 30))
        p_sim = job.get("p_sim")
        pf = max(0.0, float(p_sim)) if p_sim is not None else 0.5
        # Taste = closeness to the centroid of liked jobs minus closeness to disliked
        # ones (0.5 ~ "unrelated" for typical embedding models, so we centre there).
        taste = 0.0
        like_sim = job.get("like_sim")
        if like_sim is not None:
            taste += max(0.0, float(like_sim) - 0.5)
        if job.get("dislike_sim") is not None:
            taste -= max(0.0, float(job["dislike_sim"]) - 0.5)
        score = (0.40 * rel + 0.20 * pf + 0.20 * so + 0.08 * lf + 0.07 * ef + 0.05 * rec
                 + 0.10 * taste - 0.30 * ghost)

        floor = profile.get("min_salary_lpa")
        smax = job.get("salary_max_lpa")
        if floor and smax is not None and smax < floor:
            score -= 0.10
        if matched:
            total = len(job.get("skills") or [])
            reasons.append(f"Matches {len(matched)}/{total} skills: {', '.join(matched[:4])}")
        if loc_reason:
            reasons.append(loc_reason)
        if exp_reason:
            reasons.append(exp_reason)
        if like_sim is not None and float(like_sim) >= 0.8:
            reasons.append("Similar to jobs you liked")

    if age_days <= 7:
        reasons.append(f"Posted {age_days} day{'s' if age_days != 1 else ''} ago")
    return round(score, 4), reasons[:4], missing[:4]
