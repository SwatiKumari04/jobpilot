"""GET /health (used by Render) and GET /api/stats (usage numbers)."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ... import db

router = APIRouter(tags=["health"])
log = logging.getLogger(__name__)


@router.get("/health")
def health() -> dict:
    try:
        with db.conn() as c:
            c.execute("SELECT 1")
        return {"status": "ok", "db": True}
    except Exception:
        log.exception("health check failed")
        raise HTTPException(503, "database unavailable")


@router.get("/api/stats")
def stats() -> dict:
    with db.conn() as c:
        row = c.execute(
            """SELECT
                 (SELECT count(*) FROM jobs WHERE is_active AND duplicate_of IS NULL) AS active_jobs,
                 (SELECT count(*) FROM jobs WHERE duplicate_of IS NOT NULL)          AS duplicates_merged,
                 (SELECT count(*) FROM jobs WHERE is_active AND ghost_score >= 0.7)  AS ghost_flagged,
                 (SELECT count(*) FROM users)                                        AS users,
                 (SELECT count(*) FROM events WHERE kind = 'search')                 AS searches,
                 (SELECT count(*) FROM events WHERE kind = 'chat')                   AS chats,
                 (SELECT count(*) FROM events WHERE kind = 'digest')                 AS digests"""
        ).fetchone()
    return dict(row)
