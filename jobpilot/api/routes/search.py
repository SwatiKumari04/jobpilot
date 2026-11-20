"""GET /api/search: hybrid semantic + keyword search, personalised when user_id is given."""
from __future__ import annotations

import time

from fastapi import APIRouter, Query

from ... import db
from ...search import Filters, job_card, search
from ...services import profiles
from ..deps import embedder

router = APIRouter(tags=["search"])


@router.get("/api/search")
def search_jobs(
    q: str = Query(min_length=1, max_length=300),
    user_id: str | None = None,
    location: str | None = None,
    remote_only: bool = False,
    limit: int = Query(10, ge=1, le=30),
    mode: str = Query("hybrid", pattern="^(hybrid|vector|keyword)$"),
) -> dict:
    user = profiles.get_user(user_id) if user_id else None
    t0 = time.perf_counter()
    rows = search(q, embedder(), user=user, limit=limit, mode=mode,  # type: ignore[arg-type]
                  filters=Filters(remote_only=remote_only, location=location))
    ms = round((time.perf_counter() - t0) * 1000)
    db.log_event("search", user["id"] if user else None, {"q": q[:100], "n": len(rows), "ms": ms})
    return {"query": q, "took_ms": ms, "results": [job_card(r) for r in rows]}
