"""Likes / dislikes on jobs (used for personalisation)."""
from __future__ import annotations

from .. import db


def save_feedback(user_id: str, job_id: int, signal: int) -> None:
    with db.conn() as c:
        c.execute(
            "INSERT INTO feedback (user_id, job_id, signal) VALUES (%s, %s, %s) "
            "ON CONFLICT (user_id, job_id) DO UPDATE SET signal = EXCLUDED.signal, created_at = now()",
            (user_id, job_id, signal),
        )
