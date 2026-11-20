"""POST /api/admin/ingest: push a batch of jobs into the index (protected by ADMIN_TOKEN)."""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from fastapi.concurrency import run_in_threadpool

from ... import ingest
from ...schemas.api import IngestIn
from ..deps import embedder, settings

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/ingest")
async def admin_ingest(body: IngestIn, x_admin_token: str = Header(default="")) -> dict:
    if not settings.admin_token or settings.admin_token == "change-me" or x_admin_token != settings.admin_token:
        raise HTTPException(401, "bad admin token")
    out = await run_in_threadpool(ingest.ingest_jobs, body.jobs, embedder(), body.full_snapshot_of_source)
    return out.model_dump()
