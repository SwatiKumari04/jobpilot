"""Postgres connection pool + helpers.

We write plain SQL (no ORM) on purpose: it keeps every query visible and
shows SQL skills. pgvector values are passed as text like '[0.1,0.2,...]'
and cast with ::vector, so we don't need numpy.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from ..config import Settings

log = logging.getLogger(__name__)
_pool: ConnectionPool | None = None


def init_pool(settings: Settings) -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=5,
            # prepare_threshold=None: works behind PgBouncer / Neon's pooled endpoint.
            kwargs={"row_factory": dict_row, "prepare_threshold": None},
            check=ConnectionPool.check_connection,  # drop dead connections (Neon scales to zero)
            open=True,
        )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def conn() -> Iterator[Any]:
    if _pool is None:
        raise RuntimeError("Database pool not initialised; call init_pool() first")
    with _pool.connection() as c:
        yield c


def init_schema(settings: Settings) -> None:
    sql = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    sql = sql.replace("__EMBED_DIM__", str(settings.embed_dim))
    with conn() as c:
        c.execute(sql)
    log.info("Database schema ready (embedding dim=%s)", settings.embed_dim)


def to_vector(values: Sequence[float] | None) -> str | None:
    """Python list -> pgvector text literal."""
    if values is None:
        return None
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"


def jsonb(value: Any) -> Jsonb:
    return Jsonb(value)


def log_event(kind: str, user_id: str | None = None, meta: dict | None = None) -> None:
    try:
        with conn() as c:
            c.execute(
                "INSERT INTO events (user_id, kind, meta) VALUES (%s, %s, %s)",
                (user_id, kind, Jsonb(meta or {})),
            )
    except Exception:  # analytics must never break a request
        log.exception("could not log event %s", kind)


def dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)
