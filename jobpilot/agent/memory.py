"""Short-term memory: the last N chat turns, stored in Postgres."""
from __future__ import annotations

from .. import db

HISTORY_TURNS = 12


def load_history(user_id: str, turns: int = HISTORY_TURNS) -> list[dict]:
    with db.conn() as c:
        rows = c.execute(
            "SELECT role, content FROM messages WHERE user_id = %s ORDER BY created_at DESC, id DESC LIMIT %s",
            (user_id, turns),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def save_turn(user_id: str, user_message: str, reply: str) -> None:
    with db.conn() as c:
        c.execute("INSERT INTO messages (user_id, role, content) VALUES (%s, 'user', %s)", (user_id, user_message))
        c.execute("INSERT INTO messages (user_id, role, content) VALUES (%s, 'assistant', %s)", (user_id, reply))
