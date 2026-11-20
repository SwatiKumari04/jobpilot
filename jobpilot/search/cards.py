"""How a job is shown to the UI, the API and the LLM."""
from __future__ import annotations


def job_card(row: dict, description_chars: int = 280) -> dict:
    """Compact JSON-safe view of a job for the API, the UI and the LLM."""
    salary = None
    if row.get("salary_min_lpa") is not None or row.get("salary_max_lpa") is not None:
        lo, hi = row.get("salary_min_lpa"), row.get("salary_max_lpa")
        salary = f"{lo:g}-{hi:g} LPA" if lo is not None and hi is not None else f"{(lo or hi):g} LPA"
    desc = row.get("description") or ""
    return {
        "id": row["id"],
        "title": row["title"],
        "company": row["company"],
        "location": row.get("location") or "",
        "remote": bool(row.get("remote")),
        "level": row.get("level") or "",
        "salary": salary,
        "skills": (row.get("skills") or [])[:8],
        "posted_at": row["posted_at"].date().isoformat() if row.get("posted_at") else None,
        "url": row.get("url") or "",
        "source": row.get("source") or "",
        "score": row.get("score"),
        "reasons": row.get("reasons") or [],
        "missing_skills": row.get("missing_skills") or [],
        "ghost_score": row.get("ghost_score"),
        "snippet": desc[:description_chars] + ("..." if len(desc) > description_chars else ""),
    }
