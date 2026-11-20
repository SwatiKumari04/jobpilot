"""Ingestion pipeline for a constantly changing jobs index.

For every incoming listing:
  * normalise it and extract skills (cheap keyword matching, no LLM per job),
  * detect cross-posts of the same job (dedupe_key) -> count reposts,
  * skip re-embedding when content is unchanged (content_hash),
  * compute a ghost-listing score,
then embed only new/changed jobs and expire listings that disappeared.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .. import db
from ..llm import Embedder
from ..utils import content_hash, dedupe_key, extract_skills, ghost_score, normalize_skills, strip_html

log = logging.getLogger(__name__)


class JobIn(BaseModel):
    source: str
    external_id: str
    title: str
    company: str
    location: str = ""
    remote: bool = False
    category: str = ""
    level: str = ""
    experience_min: float = 0
    experience_max: float = 30
    salary_min_lpa: float | None = None
    salary_max_lpa: float | None = None
    skills: list[str] = Field(default_factory=list)
    description: str = ""
    url: str = ""
    posted_at: datetime | None = None


class IngestStats(BaseModel):
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    duplicates: int = 0
    embedded: int = 0
    expired: int = 0


def job_text(row: dict) -> str:
    """The text we embed for a job."""
    where = row.get("location") or ""
    if row.get("remote"):
        where = (where + " (remote)").strip()
    skills = ", ".join(row.get("skills") or [])
    return (f"{row['title']} at {row['company']}. Level: {row.get('level') or 'any'}. "
            f"Location: {where}. Skills: {skills}. {(row.get('description') or '')[:1500]}")


def _prepare(job: JobIn) -> dict:
    d = job.model_dump()
    d["description"] = strip_html(d["description"])
    d["skills"] = normalize_skills(d["skills"]) or extract_skills(f"{d['title']}\n{d['description']}")
    d["posted_at"] = d["posted_at"] or datetime.now(timezone.utc)
    if d["posted_at"].tzinfo is None:
        d["posted_at"] = d["posted_at"].replace(tzinfo=timezone.utc)
    d["dedupe_key"] = dedupe_key(d["company"], d["title"], d["location"])
    d["content_hash"] = content_hash(d["title"], d["company"], d["location"], d["remote"], d["level"],
                                     d["experience_min"], d["experience_max"], d["salary_min_lpa"],
                                     d["salary_max_lpa"], ",".join(d["skills"]), d["description"])
    return d


_FIELDS = ["title", "company", "location", "remote", "category", "level", "experience_min", "experience_max",
           "salary_min_lpa", "salary_max_lpa", "skills", "description", "url", "posted_at", "content_hash", "dedupe_key"]


def ingest_jobs(jobs: list[JobIn], embedder: Embedder, expire_missing_for_source: str | None = None) -> IngestStats:
    """Upsert a batch of listings. If expire_missing_for_source is set, this batch is
    treated as a full snapshot of that source: anything not in it gets deactivated."""
    stats = IngestStats()
    run_started = datetime.now(timezone.utc)
    with db.conn() as c:
        for job in jobs:
            d = _prepare(job)
            existing = c.execute(
                "SELECT id, content_hash, duplicate_of, repost_count FROM jobs WHERE source = %s AND external_id = %s",
                (d["source"], d["external_id"]),
            ).fetchone()

            if existing and existing["content_hash"] == d["content_hash"]:
                c.execute(
                    "UPDATE jobs SET last_seen_at = now(), is_active = (duplicate_of IS NULL) WHERE id = %s",
                    (existing["id"],),
                )
                if existing["duplicate_of"]:
                    c.execute("UPDATE jobs SET last_seen_at = now() WHERE id = %s", (existing["duplicate_of"],))
                stats.unchanged += 1
                continue

            if existing:  # content changed -> update and re-embed
                g = ghost_score(d["posted_at"], d["description"], d["salary_max_lpa"] is not None,
                                existing["repost_count"])
                sets = ", ".join(f"{f} = %({f})s" for f in _FIELDS)
                c.execute(
                    f"UPDATE jobs SET {sets}, ghost_score = %(ghost)s, embedding = NULL, last_seen_at = now(), "
                    f"is_active = (duplicate_of IS NULL) WHERE id = %(id)s",
                    {**d, "ghost": g, "id": existing["id"]},
                )
                stats.updated += 1
                continue

            canonical = c.execute(
                "SELECT id, posted_at, description, salary_max_lpa, repost_count FROM jobs "
                "WHERE dedupe_key = %s AND duplicate_of IS NULL AND is_active ORDER BY id LIMIT 1",
                (d["dedupe_key"],),
            ).fetchone()
            if canonical:  # cross-post of a job we already have
                reposts = canonical["repost_count"] + 1
                g = ghost_score(canonical["posted_at"], canonical["description"],
                                canonical["salary_max_lpa"] is not None, reposts)
                c.execute(
                    "UPDATE jobs SET repost_count = %s, ghost_score = %s, last_seen_at = now() WHERE id = %s",
                    (reposts, g, canonical["id"]),
                )
                cols = ", ".join(["source", "external_id", *_FIELDS, "is_active", "duplicate_of"])
                vals = ", ".join(f"%({f})s" for f in ["source", "external_id", *_FIELDS])
                c.execute(
                    f"INSERT INTO jobs ({cols}) VALUES ({vals}, FALSE, %(dup)s)",
                    {**d, "dup": canonical["id"]},
                )
                stats.duplicates += 1
                continue

            g = ghost_score(d["posted_at"], d["description"], d["salary_max_lpa"] is not None, 0)
            cols = ", ".join(["source", "external_id", *_FIELDS, "ghost_score"])
            vals = ", ".join(f"%({f})s" for f in ["source", "external_id", *_FIELDS, "ghost"])
            c.execute(f"INSERT INTO jobs ({cols}) VALUES ({vals})", {**d, "ghost": g})
            stats.inserted += 1

        if expire_missing_for_source:
            cur = c.execute(
                "UPDATE jobs SET is_active = FALSE WHERE source = %s AND is_active AND last_seen_at < %s",
                (expire_missing_for_source, run_started),
            )
            stats.expired += cur.rowcount or 0

    stats.embedded = embed_missing(embedder)
    log.info("ingest done: %s", stats.model_dump())
    return stats


def embed_missing(embedder: Embedder, batch_size: int = 100) -> int:
    """Embed active canonical jobs that have no vector yet (new or changed)."""
    total = 0
    while True:
        with db.conn() as c:
            rows = c.execute(
                "SELECT id, title, company, location, remote, level, skills, description FROM jobs "
                "WHERE embedding IS NULL AND is_active AND duplicate_of IS NULL ORDER BY id LIMIT %s",
                (batch_size,),
            ).fetchall()
        if not rows:
            return total
        vectors = embedder.embed([job_text(r) for r in rows], kind="document")
        with db.conn() as c:
            for r, v in zip(rows, vectors):
                c.execute("UPDATE jobs SET embedding = %s::vector WHERE id = %s", (db.to_vector(v), r["id"]))
        total += len(rows)


def expire_old(max_age_days: int = 90) -> int:
    with db.conn() as c:
        cur = c.execute(
            "UPDATE jobs SET is_active = FALSE WHERE is_active AND posted_at < now() - make_interval(days => %s)",
            (max_age_days,),
        )
        return cur.rowcount or 0


def refresh_ghost_scores() -> int:
    """Listings age every day, so ghost scores must be recomputed (run nightly)."""
    with db.conn() as c:
        rows = c.execute(
            "SELECT id, posted_at, description, salary_max_lpa, repost_count FROM jobs WHERE is_active"
        ).fetchall()
        for r in rows:
            g = ghost_score(r["posted_at"], r["description"], r["salary_max_lpa"] is not None, r["repost_count"])
            c.execute("UPDATE jobs SET ghost_score = %s WHERE id = %s", (g, r["id"]))
    return len(rows)
