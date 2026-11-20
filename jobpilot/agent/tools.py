"""The tools the agent can call, and the code that runs them for one user."""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field, ValidationError

from .. import db
from ..llm import Embedder
from ..schemas.api import PreferencesIn
from ..search import Filters, job_card, search
from ..services import profiles
from ..services.feedback import save_feedback

log = logging.getLogger(__name__)

# Tool schemas are written by hand (simple JSON Schema) because some providers
# reject the anyOf/null constructs Pydantic generates for optional fields.
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Search the live jobs index (semantic + keyword), personalised for this user. "
                           "Returns the best matching jobs with match reasons and missing skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, e.g. 'python backend internship'"},
                    "location": {"type": "string", "description": "City to filter on, e.g. 'Bengaluru'. Remote jobs are always included."},
                    "remote_only": {"type": "boolean", "description": "Only remote jobs"},
                    "max_experience_years": {"type": "number", "description": "Only jobs whose minimum experience is at most this"},
                    "limit": {"type": "integer", "description": "How many results (1-10, default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_details",
            "description": "Get the full description of one job by id.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "integer"}},
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_preferences",
            "description": "Save lasting preferences the user stated (long-term memory). Only include fields the user mentioned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_roles": {"type": "array", "items": {"type": "string"}},
                    "locations": {"type": "array", "items": {"type": "string"}},
                    "remote_ok": {"type": "boolean"},
                    "min_salary_lpa": {"type": "number", "description": "Minimum salary in lakhs per annum"},
                    "extra_skills": {"type": "array", "items": {"type": "string"}, "description": "Skills to add to the profile"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_job",
            "description": "Record that the user likes (liked=true) or rejects (liked=false) a job.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "integer"}, "liked": {"type": "boolean"}},
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_saved_jobs",
            "description": "List jobs the user has saved / liked.",
            # Gemini rejects OBJECT schemas with empty `properties`, so give it one optional field.
            "parameters": {"type": "object", "properties": {
                "limit": {"type": "integer", "description": "Max jobs to return (default 20)"}}},
        },
    },
]


class SearchArgs(BaseModel):
    query: str = Field(min_length=1)
    location: str | None = None
    remote_only: bool = False
    max_experience_years: float | None = None
    limit: int = Field(default=5, ge=1, le=10)


class JobIdArgs(BaseModel):
    job_id: int


class SaveArgs(BaseModel):
    job_id: int
    liked: bool = True


class ToolBox:
    """Executes tool calls for one user. Keeps every job it showed (for grounding checks)."""

    def __init__(self, user: dict, embedder: Embedder):
        self.user = user
        self.embedder = embedder
        self.shown: dict[int, dict] = {}

    def run(self, name: str, raw_args: str) -> dict:
        try:
            args = json.loads(raw_args or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return {"error": f"unknown tool '{name}'"}
            return handler(args)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            # Returned to the model so it can fix its own call.
            return {"error": f"invalid arguments for {name}: {exc}"[:800]}
        except Exception as exc:
            log.exception("tool %s failed", name)
            return {"error": f"{name} failed: {type(exc).__name__}"}

    def _tool_search_jobs(self, args: dict) -> dict:
        a = SearchArgs.model_validate(args)
        rows = search(
            a.query, self.embedder, user=self.user, limit=a.limit,
            filters=Filters(remote_only=a.remote_only, location=a.location, max_experience_years=a.max_experience_years),
        )
        cards = [job_card(r, description_chars=200) for r in rows]
        for card in cards:
            self.shown[card["id"]] = card
        slim = [{k: c[k] for k in ("id", "title", "company", "location", "remote", "level", "salary",
                                    "skills", "reasons", "missing_skills", "posted_at", "snippet")} for c in cards]
        return {"count": len(slim), "jobs": slim}

    def _tool_get_job_details(self, args: dict) -> dict:
        a = JobIdArgs.model_validate(args)
        with db.conn() as c:
            row = c.execute(
                "SELECT id, source, title, company, location, remote, level, experience_min, experience_max, "
                "salary_min_lpa, salary_max_lpa, skills, description, url, posted_at, ghost_score, repost_count "
                "FROM jobs WHERE id = %s",
                (a.job_id,),
            ).fetchone()
        if not row:
            return {"error": f"job {a.job_id} not found"}
        card = job_card(row, description_chars=2000)
        self.shown[card["id"]] = card
        card["experience"] = f"{row['experience_min']:g}-{row['experience_max']:g} years"
        card["ghost_warning"] = row["ghost_score"] >= 0.5
        return card

    def _tool_update_preferences(self, args: dict) -> dict:
        prefs = PreferencesIn.model_validate(args)
        self.user = profiles.update_preferences(self.embedder, self.user["id"], prefs)
        return {"ok": True, "now_remembered": self.user["effective_profile"]}

    def _tool_save_job(self, args: dict) -> dict:
        a = SaveArgs.model_validate(args)
        save_feedback(self.user["id"], a.job_id, 1 if a.liked else -1)
        return {"ok": True, "job_id": a.job_id, "liked": a.liked}

    def _tool_get_saved_jobs(self, args: dict) -> dict:
        with db.conn() as c:
            rows = c.execute(
                "SELECT j.id, j.source, j.title, j.company, j.location, j.remote, j.level, j.salary_min_lpa, "
                "j.salary_max_lpa, j.skills, j.description, j.url, j.posted_at, j.ghost_score "
                "FROM feedback f JOIN jobs j ON j.id = f.job_id WHERE f.user_id = %s AND f.signal = 1 "
                "ORDER BY f.created_at DESC LIMIT %s",
                (self.user["id"], max(1, min(50, int(args.get("limit") or 20)))),
            ).fetchall()
        cards = [job_card(r, description_chars=120) for r in rows]
        for card in cards:
            self.shown[card["id"]] = card
        return {"count": len(cards), "jobs": [{k: c[k] for k in ("id", "title", "company", "location")} for c in cards]}
