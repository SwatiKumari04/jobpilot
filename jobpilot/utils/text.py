"""Text helpers for listings: HTML cleanup, dedupe keys, content hashes, ghost-listing score."""
from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime, timezone


def strip_html(text: str) -> str:
    text = re.sub(r"<(script|style).*?</\1>", " ", text or "", flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


_TITLE_NOISE = re.compile(r"\b(urgent|hiring|immediate joiner|wfh|remote|hybrid|\d+\s*(yrs?|years?))\b")


def dedupe_key(company: str, title: str, location: str) -> str:
    """Same job cross-posted on several boards -> same key."""
    t = _TITLE_NOISE.sub(" ", _norm(title))
    key = f"{_norm(company)}|{' '.join(t.split())}|{_norm(location)}"
    return hashlib.sha1(key.encode()).hexdigest()


def content_hash(*parts: object) -> str:
    return hashlib.sha256("␟".join("" if p is None else str(p) for p in parts).encode()).hexdigest()


_VAGUE = re.compile(r"(always hiring|talent pool|multiple openings|future opportunities|general application|evergreen)", re.I)


def ghost_score(posted_at: datetime, description: str, has_salary: bool, repost_count: int,
                now: datetime | None = None) -> float:
    """Heuristic 0..1 score for 'ghost' listings (stale, vague, endlessly reposted).

    Elip highlights ghost-job detection as a product feature; this is a simple,
    explainable first version. A learned model could replace it later.
    """
    now = now or datetime.now(timezone.utc)
    age_days = max(0.0, (now - posted_at).total_seconds() / 86400)
    score = 0.0
    if age_days > 30:
        score += 0.35
    if age_days > 60:
        score += 0.2
    if not has_salary:
        score += 0.1
    if len(description or "") < 300:
        score += 0.2
    if repost_count >= 3:
        score += 0.2
    if _VAGUE.search(description or ""):
        score += 0.25
    return round(min(1.0, score), 3)


def ghost_reasons(posted_at: datetime, description: str, has_salary: bool, repost_count: int,
                  now: datetime | None = None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    age_days = (now - posted_at).total_seconds() / 86400
    reasons = []
    if age_days > 30:
        reasons.append(f"posted {int(age_days)} days ago")
    if repost_count >= 3:
        reasons.append(f"reposted {repost_count} times")
    if _VAGUE.search(description or ""):
        reasons.append("vague 'always hiring' wording")
    if len(description or "") < 300:
        reasons.append("very short description")
    if not has_salary:
        reasons.append("no salary listed")
    return reasons
