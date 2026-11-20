"""Daily digest: "your 3 best jobs today" (Elip sends exactly this every morning).

retrieve + rank (cheap, deterministic)  ->  LLM picks & explains top 3 (structured output)
->  validate against the candidate list (grounding)  ->  fall back to pure ranking if needed.
"""
from __future__ import annotations

import json
import logging
import time

from pydantic import ValidationError

from ..llm import ChatLLM, Embedder, StructuredOutputError, prompts
from ..schemas import Digest
from ..search import job_card, search
from .profiles import profile_query

log = logging.getLogger(__name__)
N_CANDIDATES = 10
N_PICKS = 3


def validate_digest(d: Digest, candidate_ids: set[int], n: int) -> list[str]:
    """Return a list of problems (empty = valid). This is the hallucination guard."""
    problems = []
    ids = [p.job_id for p in d.picks]
    if len(ids) != n:
        problems.append(f"expected exactly {n} picks, got {len(ids)}")
    bad = [i for i in ids if i not in candidate_ids]
    if bad:
        problems.append(f"job ids not in the candidate list: {bad}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate job ids")
    for p in d.picks:
        if not p.why:
            problems.append(f"pick {p.job_id} has no reasons")
    return problems


def _fallback(cards: list[dict], n: int) -> dict:
    picks = []
    for card in cards[:n]:
        picks.append({
            "job": card,
            "fit_score": int(round(max(0.0, min(1.0, card["score"] or 0)) * 100)),
            "why": card["reasons"] or ["Strong match for your search profile"],
            "missing_skills": card["missing_skills"],
        })
    return {"summary": "Top matches from the ranking system.", "picks": picks}


def build_digest(llm: ChatLLM, embedder: Embedder, user: dict) -> dict:
    t0 = time.perf_counter()
    profile = user["effective_profile"]
    rows = search(profile_query(profile), embedder, user=user, limit=N_CANDIDATES)
    cards = [job_card(r, description_chars=400) for r in rows]
    n = min(N_PICKS, len(cards))
    meta = {"source": "llm", "candidates": len(cards), "attempts": 0, "first_try_valid": False,
            "prompt_version": prompts.PROMPT_VERSION}
    if n == 0:
        return {"summary": "No matching jobs right now. Try adding skills or broader roles.", "picks": [], "meta": meta}

    by_id = {c["id"]: c for c in cards}
    jobs_txt = "\n".join(
        json.dumps({k: c[k] for k in ("id", "title", "company", "location", "remote", "level", "salary", "skills", "snippet")},
                   ensure_ascii=False)
        for c in cards
    )
    messages = [{"role": "user", "content": prompts.DIGEST.format(profile=json.dumps(profile, ensure_ascii=False), jobs=jobs_txt)}]
    for attempt in range(2):  # one repair round if the output is not grounded
        meta["attempts"] += 1
        try:
            digest, _stats = llm.structured(messages, Digest)
        except (StructuredOutputError, ValidationError, Exception) as exc:  # noqa: BLE001 - degrade gracefully
            log.warning("digest LLM call failed: %s", exc)
            break
        problems = validate_digest(digest, set(by_id), n)
        if not problems:
            meta["first_try_valid"] = attempt == 0
            meta["ms"] = round((time.perf_counter() - t0) * 1000)
            return {
                "summary": digest.summary,
                "picks": [{"job": by_id[p.job_id], "fit_score": p.fit_score, "why": p.why[:3],
                           "missing_skills": p.missing_skills[:4]} for p in digest.picks],
                "meta": meta,
            }
        log.warning("digest invalid (attempt %s): %s", attempt + 1, problems)
        messages = messages + [
            {"role": "assistant", "content": digest.model_dump_json()},
            {"role": "user", "content": "Your answer has problems: " + "; ".join(problems) + ". Fix them and answer again."},
        ]
    out = _fallback(cards, n)
    out["meta"] = {**meta, "source": "fallback", "ms": round((time.perf_counter() - t0) * 1000)}
    return out
