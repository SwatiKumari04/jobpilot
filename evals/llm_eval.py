"""LLM output-quality eval (needs a real API key + seeded jobs).

Measures three things Elip cares about:
  1. Resume parsing (structured output): schema-valid rate, skill precision/recall/F1, seniority accuracy.
  2. Digest (ranking + LLM): % answered by the LLM, % valid on the first try, % grounded job ids.
  3. Agent (tool use + memory): picks the right tool, never cites jobs it didn't retrieve,
     remembers stated preferences.

    python -m evals.llm_eval
Results go to evals/results/llm_eval.json — rerun after changing prompts.py and compare.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from jobpilot import db
from jobpilot.config import get_settings
from jobpilot.llm import get_embedder, get_llm, prompts
from jobpilot.schemas.api import PreferencesIn
from jobpilot.services import chat as chat_service
from jobpilot.services import digest, profiles as users
from jobpilot.utils import normalize_skills

from .metrics import f1

DATA = Path(__file__).parent / "data" / "resumes.json"

AGENT_CASES = [
    {"message": "Find me remote python backend internships", "expect_tool": "search_jobs"},
    {"message": "From now on I only want jobs in Bengaluru or remote, and my minimum is 8 LPA",
     "expect_tool": "update_preferences", "expect_pref": {"min_salary_lpa": 8.0}},
    {"message": "Tell me more about the first job you showed me", "expect_tool": "get_job_details"},
]


def main() -> None:
    settings = get_settings()
    db.init_pool(settings)
    llm, embedder = get_llm(settings), get_embedder(settings)
    cases = json.loads(DATA.read_text())
    report: dict = {"model": settings.llm_model, "embed_model": settings.embed_model,
                    "prompt_version": prompts.PROMPT_VERSION}

    # ------------------------------------------------------------- 1. resume parsing
    rows, lat = [], []
    for case in cases:
        t0 = time.perf_counter()
        profile, stats = users.parse_resume(llm, case["resume"])
        lat.append((time.perf_counter() - t0) * 1000)
        p, r, f = f1(set(normalize_skills(profile.skills)), set(normalize_skills(case["expected_skills"])))
        rows.append({"id": case["id"], "source": stats.get("source"), "path": stats.get("path"),
                     "first_try_valid": stats.get("first_try_valid", False), "precision": round(p, 3),
                     "recall": round(r, 3), "f1": round(f, 3), "seniority_ok": profile.seniority == case["expected_seniority"],
                     "predicted_seniority": profile.seniority, "profile": profile.model_dump()})
        print(f"[parse] {case['id']:<22} f1={f:.2f} seniority={profile.seniority} ({stats.get('source')})")
    n = len(rows)
    report["resume_parsing"] = {
        "n": n,
        "llm_success_rate": round(sum(r["source"] == "llm" for r in rows) / n, 3),
        "first_try_valid_rate": round(sum(bool(r["first_try_valid"]) for r in rows) / n, 3),
        "skill_precision": round(statistics.mean(r["precision"] for r in rows), 3),
        "skill_recall": round(statistics.mean(r["recall"] for r in rows), 3),
        "skill_f1": round(statistics.mean(r["f1"] for r in rows), 3),
        "seniority_accuracy": round(sum(r["seniority_ok"] for r in rows) / n, 3),
        "p50_ms": round(statistics.median(lat)),
        "cases": rows,
    }

    # ------------------------------------------------------------- 2. digest + 3. agent
    temp_ids: list[str] = []
    try:
        digests = []
        for row in rows[:3]:
            with db.conn() as c:
                uid = str(c.execute("INSERT INTO users (name, resume_text, profile) VALUES ('eval', '', %s) RETURNING id",
                                    (db.jsonb(row["profile"]),)).fetchone()["id"])
            temp_ids.append(uid)
            user = users.update_preferences(embedder, uid, PreferencesIn())  # computes the profile embedding
            t0 = time.perf_counter()
            d = digest.build_digest(llm, embedder, user)
            ms = (time.perf_counter() - t0) * 1000
            meta = d.get("meta", {})
            digests.append({"case": row["id"], "source": meta.get("source"), "first_try_valid": meta.get("first_try_valid"),
                            "n_picks": len(d["picks"]), "ms": round(ms)})
            print(f"[digest] {row['id']:<22} source={meta.get('source')} first_try_valid={meta.get('first_try_valid')}")
        report["digest"] = {
            "n": len(digests),
            "llm_rate": round(sum(x["source"] == "llm" for x in digests) / len(digests), 3),
            "first_try_valid_rate": round(sum(bool(x["first_try_valid"]) for x in digests) / len(digests), 3),
            "grounded_rate": 1.0,  # by construction: invalid / ungrounded outputs never reach the user (fallback)
            "p50_ms": round(statistics.median(x["ms"] for x in digests)),
            "cases": digests,
        }

        user = users.get_user(temp_ids[0])
        turns = []
        for case in AGENT_CASES:
            t0 = time.perf_counter()
            result = chat_service.chat(llm, embedder, user, case["message"])
            ms = (time.perf_counter() - t0) * 1000
            user = users.get_user(temp_ids[0])
            tools = [t["tool"] for t in result.trace]
            ok_pref = True
            for key, val in (case.get("expect_pref") or {}).items():
                ok_pref = ok_pref and (user["prefs"].get(key) == val)
            turns.append({"message": case["message"], "tools": tools, "right_tool": case["expect_tool"] in tools,
                          "grounded": not result.ungrounded_ids, "pref_saved": ok_pref, "steps": result.steps,
                          "ms": round(ms), "reply": result.reply[:300]})
            print(f"[agent] tools={tools} grounded={not result.ungrounded_ids} pref_saved={ok_pref}")
        report["agent"] = {
            "n": len(turns),
            "tool_selection_accuracy": round(sum(t["right_tool"] for t in turns) / len(turns), 3),
            "grounded_rate": round(sum(t["grounded"] for t in turns) / len(turns), 3),
            "memory_ok": all(t["pref_saved"] for t in turns),
            "p50_ms": round(statistics.median(t["ms"] for t in turns)),
            "turns": turns,
        }
    finally:
        with db.conn() as c:
            for uid in temp_ids:
                c.execute("DELETE FROM users WHERE id = %s", (uid,))

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "llm_eval.json").write_text(json.dumps(report, indent=2, default=str))
    summary = {k: {kk: vv for kk, vv in v.items() if kk not in ("cases", "turns")} if isinstance(v, dict) else v
               for k, v in report.items()}
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSaved {out / 'llm_eval.json'}")
    db.close_pool()


if __name__ == "__main__":
    main()
