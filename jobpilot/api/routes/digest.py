"""GET /api/users/{id}/digest: today's top 3 jobs, picked and explained by the LLM."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from ... import db
from ...services import digest as digest_service
from ..deps import embedder, llm, llm_rate_limit, require_user

router = APIRouter(tags=["digest"])


@router.get("/api/users/{user_id}/digest", dependencies=[Depends(llm_rate_limit)])
async def get_digest(user_id: str) -> dict:
    user = require_user(user_id)
    out = await run_in_threadpool(digest_service.build_digest, llm(), embedder(), user)
    db.log_event("digest", user["id"], out.get("meta"))
    return out
