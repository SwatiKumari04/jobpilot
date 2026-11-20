"""Users: create from a resume (LLM structured output), read, update preferences."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from ... import db
from ...schemas.api import CreateUserIn, PreferencesIn
from ...services import profiles
from ..deps import embedder, llm, llm_rate_limit, require_user

router = APIRouter(prefix="/api/users", tags=["users"])


@router.post("", dependencies=[Depends(llm_rate_limit)])
async def create_user(body: CreateUserIn) -> dict:
    user = await run_in_threadpool(profiles.create_user, llm(), embedder(), body.name, body.resume_text)
    db.log_event("signup", user["id"], {"parse": user.get("parse_stats")})
    return profiles.public_user(user)


@router.get("/{user_id}")
def get_user(user_id: str) -> dict:
    return profiles.public_user(require_user(user_id))


@router.patch("/{user_id}/preferences")
def update_preferences(user_id: str, body: PreferencesIn) -> dict:
    require_user(user_id)
    return profiles.public_user(profiles.update_preferences(embedder(), user_id, body))
