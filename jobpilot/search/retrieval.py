"""Hybrid semantic + keyword search with personalised re-ranking.

1. Keyword retrieval  : Postgres full-text search (tsvector + ts_rank_cd).
2. Semantic retrieval : pgvector cosine distance on an HNSW index.
3. Fusion             : Reciprocal Rank Fusion of the two ranked lists.
4. Re-rank            : personal signals (profile, skills, location, experience,
                        recency, likes/dislikes, ghost-listing penalty).
"""
from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

from .. import db
from ..llm import Embedder
from .ranking import rrf_fuse, score_job

Mode = Literal["hybrid", "vector", "keyword"]
GHOST_CUTOFF = 0.7


@dataclass
class Filters:
    remote_only: bool = False
    location: str | None = None
    max_experience_years: float | None = None
    sources: list[str] | None = None
    include_ghost: bool = False
    exclude_job_ids: list[int] = field(default_factory=list)


def _where(f: Filters) -> tuple[str, dict]:
    parts = ["j.is_active", "j.duplicate_of IS NULL", "j.embedding IS NOT NULL"]
    params: dict = {}
    if not f.include_ghost:
        parts.append("j.ghost_score < %(ghost_cutoff)s")
        params["ghost_cutoff"] = GHOST_CUTOFF
    if f.remote_only:
        parts.append("j.remote")
    if f.location:
        parts.append("(j.location ILIKE %(loc)s OR j.remote)")
        params["loc"] = f"%{f.location}%"
    if f.max_experience_years is not None:
        parts.append("j.experience_min <= %(max_exp)s")
        params["max_exp"] = f.max_experience_years
    if f.sources:
        parts.append("j.source = ANY(%(sources)s)")
        params["sources"] = f.sources
    if f.exclude_job_ids:
        parts.append("NOT (j.id = ANY(%(exclude)s))")
        params["exclude"] = f.exclude_job_ids
    return " AND ".join(parts), params


def _or_tsquery_terms(text: str) -> str | None:
    """'python backend, bangalore' -> 'python | backend | bangalore' (OR, ranked by ts_rank_cd)."""
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1][:16]
    return " | ".join(words) if words else None


def keyword_ids(c, query: str, f: Filters, k: int) -> list[int]:
    terms = _or_tsquery_terms(query)
    if not terms:
        return []
    where, params = _where(f)
    rows = c.execute(
        f"""SELECT j.id FROM jobs j, to_tsquery('english', %(q)s) q
            WHERE {where} AND j.tsv @@ q
            ORDER BY ts_rank_cd(j.tsv, q) DESC, j.posted_at DESC
            LIMIT %(k)s""",
        {**params, "q": terms, "k": k},
    ).fetchall()
    return [r["id"] for r in rows]


def vector_ids(c, qvec: list[float], f: Filters, k: int) -> list[int]:
    where, params = _where(f)
    rows = c.execute(
        f"""SELECT j.id FROM jobs j WHERE {where}
            ORDER BY j.embedding <=> %(v)s::vector
            LIMIT %(k)s""",
        {**params, "v": db.to_vector(qvec), "k": k},
    ).fetchall()
    return [r["id"] for r in rows]


JOB_COLUMNS = ("j.id, j.source, j.title, j.company, j.location, j.remote, j.category, j.level, j.experience_min, "
               "j.experience_max, j.salary_min_lpa, j.salary_max_lpa, j.skills, j.description, j.url, j.posted_at, "
               "j.ghost_score, j.repost_count")


def fetch_jobs(c, ids: list[int], user_id: str | None = None, profile_vec: list[float] | None = None) -> dict[int, dict]:
    if not ids:
        return {}
    rows = c.execute(
        f"""WITH likes AS (
                SELECT AVG(j.embedding) AS v FROM feedback f JOIN jobs j ON j.id = f.job_id
                WHERE f.user_id = %(uid)s AND f.signal = 1),
            dislikes AS (
                SELECT AVG(j.embedding) AS v FROM feedback f JOIN jobs j ON j.id = f.job_id
                WHERE f.user_id = %(uid)s AND f.signal = -1)
            SELECT {JOB_COLUMNS},
                   1 - (j.embedding <=> %(p)s::vector)            AS p_sim,
                   1 - (j.embedding <=> (SELECT v FROM likes))    AS like_sim,
                   1 - (j.embedding <=> (SELECT v FROM dislikes)) AS dislike_sim
            FROM jobs j WHERE j.id = ANY(%(ids)s)""",
        {"uid": user_id, "p": db.to_vector(profile_vec), "ids": ids},
    ).fetchall()
    return {r["id"]: r for r in rows}


_QCACHE: OrderedDict[tuple, list[float]] = OrderedDict()
_QCACHE_LOCK = threading.Lock()
_QCACHE_SIZE = 512


def query_vector(embedder: Embedder, query: str) -> list[float]:
    """LRU cache for query embeddings: repeated searches cost no API calls."""
    key = (id(embedder), query.strip().lower())
    with _QCACHE_LOCK:
        if key in _QCACHE:
            _QCACHE.move_to_end(key)
            return _QCACHE[key]
    vec = embedder.embed([query], kind="query")[0]
    with _QCACHE_LOCK:
        _QCACHE[key] = vec
        if len(_QCACHE) > _QCACHE_SIZE:
            _QCACHE.popitem(last=False)
    return vec


def search(
    query: str,
    embedder: Embedder,
    *,
    user: dict | None = None,
    filters: Filters | None = None,
    limit: int = 10,
    mode: Mode = "hybrid",
    candidate_k: int = 50,
    personalize: bool = True,
) -> list[dict]:
    """Return ranked job dicts with `score`, `reasons`, `missing_skills`."""
    f = filters or Filters()
    user_id = str(user["id"]) if user else None
    if user and personalize:
        with db.conn() as c:
            disliked = c.execute(
                "SELECT job_id FROM feedback WHERE user_id = %s AND signal = -1", (user_id,)
            ).fetchall()
        f.exclude_job_ids = list({*f.exclude_job_ids, *[r["job_id"] for r in disliked]})

    qvec = None
    if mode in ("hybrid", "vector"):
        try:
            qvec = query_vector(embedder, query)
        except Exception:  # embedding API down / quota hit: degrade to keyword search
            if mode == "vector":
                raise
            logging.getLogger(__name__).exception("query embedding failed; keyword-only search")
    with db.conn() as c:
        lists: list[list[int]] = []
        if mode in ("hybrid", "keyword"):
            lists.append(keyword_ids(c, query, f, candidate_k))
        if mode in ("hybrid", "vector") and qvec is not None:
            lists.append(vector_ids(c, qvec, f, candidate_k))
        fused = rrf_fuse(lists)
        if not fused:
            return []
        ranked_ids = sorted(fused, key=fused.get, reverse=True)[:candidate_k]
        profile_vec = user.get("profile_embedding_list") if (user and personalize) else None
        rows = fetch_jobs(c, ranked_ids, user_id if personalize else None, profile_vec)

    top = max(fused.values())
    profile = (user.get("effective_profile") if (user and personalize) else None)
    results = []
    for jid in ranked_ids:
        row = rows.get(jid)
        if row is None:
            continue
        rel = fused[jid] / top
        score, reasons, missing = score_job(row, profile, rel)
        results.append({**row, "score": score, "relevance": round(rel, 4), "reasons": reasons, "missing_skills": missing})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]
