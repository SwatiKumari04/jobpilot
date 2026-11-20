"""POST /api/feedback: like (+1) or hide (-1) a job; drives personalisation."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ... import db
from ...schemas.api import FeedbackIn
from ...services.feedback import save_feedback
from ..deps import require_user

router = APIRouter(tags=["feedback"])


@router.post("/api/feedback")
def feedback(body: FeedbackIn) -> dict:
    require_user(body.user_id)
    try:
        save_feedback(body.user_id, body.job_id, body.signal)
    except Exception:
        raise HTTPException(404, "job not found")
    db.log_event("feedback", body.user_id, {"job_id": body.job_id, "signal": body.signal})
    return {"ok": True}
