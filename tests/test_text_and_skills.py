"""Skill extraction, dedupe keys, hashing, ghost score, HTML cleanup, embedding helpers."""
from datetime import datetime, timedelta, timezone

from jobpilot.llm import HashEmbedder, fit_dim
from jobpilot.utils import content_hash, dedupe_key, extract_skills, ghost_score, normalize_skills, strip_html

NOW = datetime(2026, 11, 20, tzinfo=timezone.utc)


def test_extract_skills_handles_aliases_and_longest_match():
    text = "We use Python, FastAPI, Postgres, React Native and k8s. JavaScript is a plus."
    skills = extract_skills(text)
    assert {"python", "fastapi", "postgresql", "react native", "kubernetes", "javascript"} <= set(skills)
    assert "react" not in skills  # 'react native' wins over 'react'
    assert "java" not in skills   # 'javascript' must not count as java


def test_ambiguous_words_need_tech_context():
    assert "go" not in extract_skills("Let's go to the office")
    assert "go" in extract_skills("Skills: Go, Python, SQL")


def test_normalize_skills_dedupes_and_maps_aliases():
    assert normalize_skills(["Postgres", "postgresql", "K8s", "Python3"]) == ["postgresql", "kubernetes", "python"]


def test_dedupe_key_ignores_noise():
    a = dedupe_key("Nimbus Labs", "Backend Engineer - URGENT HIRING", "Bengaluru")
    b = dedupe_key("nimbus labs", "Backend Engineer", "bengaluru")
    assert a == b
    assert a != dedupe_key("Nimbus Labs", "Frontend Engineer", "Bengaluru")


def test_content_hash_changes_with_content():
    assert content_hash("a", 1, None) == content_hash("a", 1, None)
    assert content_hash("a", 1) != content_hash("a", 2)


def test_ghost_score():
    fresh = ghost_score(NOW - timedelta(days=2), "x" * 500, True, 0, now=NOW)
    stale = ghost_score(NOW - timedelta(days=90), "We are always hiring for our talent pool", False, 4, now=NOW)
    assert fresh == 0.0
    assert stale >= 0.9


def test_strip_html():
    assert strip_html("<p>Hello&nbsp;<b>world</b></p><script>x()</script>") == "Hello world"


def test_fit_dim_truncates_and_normalises():
    v = fit_dim([3.0, 4.0, 100.0], 2)
    assert v == [0.6, 0.8]


def test_hash_embedder_is_deterministic_and_unit_length():
    e = HashEmbedder(64)
    a, b = e.embed(["python backend", "python backend"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
