"""POST /api/chat: talk to the tool-using agent."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from ... import db
from ...schemas.api import ChatIn
from ...services import chat as chat_service
from ..deps import embedder, llm, llm_rate_limit, require_user

router = APIRouter(tags=["agent"])
log = logging.getLogger(__name__)


@router.post("/api/chat", dependencies=[Depends(llm_rate_limit)])
async def chat(body: ChatIn) -> dict:
    user = require_user(body.user_id)
    try:
        result = await run_in_threadpool(chat_service.chat, llm(), embedder(), user, body.message)
    except Exception as exc:
        log.exception("agent failed")
        raise HTTPException(502, f"The AI model is unavailable right now ({type(exc).__name__}). Try again shortly.")
    db.log_event("chat", user["id"], {"steps": result.steps, "tools": [t["tool"] for t in result.trace],
                                      "ungrounded": result.ungrounded_ids})
    return {"reply": result.reply, "jobs": result.jobs, "trace": result.trace}
