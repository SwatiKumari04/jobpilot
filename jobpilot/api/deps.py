"""Shared dependencies for the routes: settings, clients, rate limiting, user lookup."""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from ..config import get_settings
from ..llm import ChatLLM, Embedder, get_embedder, get_llm
from ..services import profiles

settings = get_settings()


def llm() -> ChatLLM:
    return get_llm(settings)


def embedder() -> Embedder:
    return get_embedder(settings)


class SlidingWindowLimiter:
    """Per-client sliding window (protects the free LLM quota from abuse)."""

    def __init__(self, limit: int, window_s: float):
        self.limit, self.window = limit, window_s
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


limiter = SlidingWindowLimiter(settings.rate_limit_per_10min, 600)


def llm_rate_limit(request: Request) -> None:
    """Use on routes that call the LLM."""
    client = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    if not limiter.allow(client):
        raise HTTPException(429, "Too many AI requests from you; please wait a few minutes.")


def require_user(user_id: str) -> dict:
    user = profiles.get_user(user_id)
    if user is None:
        raise HTTPException(404, "user not found")
    return user
