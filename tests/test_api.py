"""End-to-end API test against a real Postgres + pgvector (no LLM API needed).

Runs only when TEST_DATABASE_URL is set (CI starts a pgvector container):
    TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/jobpilot_test pytest
WARNING: it wipes the tables in that database.
"""
import json
import os
import re

import pytest

TEST_DB = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL not set")

if TEST_DB:
    os.environ.update({"DATABASE_URL": TEST_DB, "EMBED_PROVIDER": "hash", "ADMIN_TOKEN": "test-admin",
                       "RATE_LIMIT_PER_10MIN": "1000"})

PROFILE = {"headline": "EE student into backend", "skills": ["python", "fastapi", "sql", "docker"],
           "target_roles": ["Backend Engineer Intern"], "years_experience": 0, "seniority": "intern",
           "locations": ["Bengaluru"], "education": "B.Tech EE"}


def pick_first_three(messages, schema):
    ids = [int(x) for x in re.findall(r'"id": (\d+)', messages[0]["content"])][:3]
    return {"summary": "Three good ones.", "picks": [
        {"job_id": i, "fit_score": 80, "why": ["uses your python skills"], "missing_skills": []} for i in ids]}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from fakes import ScriptedLLM

    from jobpilot import db, llm
    from jobpilot.api import main
    from jobpilot.api.deps import settings

    llm.set_embedder(llm.HashEmbedder(settings.embed_dim))
    fake = ScriptedLLM()
    llm.set_llm(fake)
    with TestClient(main.app) as c:
        with db.conn() as conn:
            conn.execute("TRUNCATE events, messages, feedback, users, jobs RESTART IDENTITY CASCADE")
        c.fake_llm = fake  # type: ignore[attr-defined]
        yield c
    llm.set_llm(None)
    llm.set_embedder(None)


def _ingest(client, jobs):
    body = {"jobs": [json.loads(j.model_dump_json()) for j in jobs]}
    return client.post("/api/admin/ingest", json=body, headers={"x-admin-token": "test-admin"})


def test_full_flow(client):
    from jobpilot.ingest.sources.synthetic import generate

    assert client.get("/health").json() == {"status": "ok", "db": True}
    jobs = generate(90, seed=1)
    assert _ingest(client, jobs[:3]).status_code == 200
    assert client.post("/api/admin/ingest", json={"jobs": []}, headers={"x-admin-token": "nope"}).status_code == 401

    stats = _ingest(client, jobs).json()
    assert stats["inserted"] == 87 and stats["unchanged"] == 3 and stats["duplicates"] == 4
    again = _ingest(client, jobs).json()  # nothing changed -> nothing re-embedded
    assert again["unchanged"] == len(jobs) and again["embedded"] == 0

    # signup: LLM parses the resume (structured output)
    client.fake_llm.structured_replies.append(PROFILE)
    user = client.post("/api/users", json={"name": "Swati", "resume_text": "B.Tech EE, Python FastAPI SQL Docker " * 3}).json()
    uid = user["id"]
    assert user["profile"]["skills"] == ["python", "fastapi", "sql", "docker"]

    # personalised hybrid search
    res = client.get("/api/search", params={"q": "python backend fastapi", "user_id": uid}).json()["results"]
    assert res and res[0]["reasons"]
    assert all(r["source"] == "synthetic" for r in res)

    # dislike the top result -> it disappears from the next search
    top = res[0]["id"]
    assert client.post("/api/feedback", json={"user_id": uid, "job_id": top, "signal": -1}).json() == {"ok": True}
    res2 = client.get("/api/search", params={"q": "python backend fastapi", "user_id": uid}).json()["results"]
    assert top not in [r["id"] for r in res2]

    # agent: tool call -> grounded answer; memory is stored
    from fakes import tool_call
    client.fake_llm.chat_replies = [
        tool_call("search_jobs", {"query": "python backend internship", "limit": 3}),
        {"role": "assistant", "content": "Here you go."},
    ]
    chat = client.post("/api/chat", json={"user_id": uid, "message": "find me backend internships"}).json()
    assert chat["reply"] == "Here you go." and chat["trace"][0]["tool"] == "search_jobs" and chat["jobs"]

    client.fake_llm.chat_replies = [
        tool_call("update_preferences", {"locations": ["Pune"], "min_salary_lpa": 6}),
        {"role": "assistant", "content": "Saved."},
    ]
    client.post("/api/chat", json={"user_id": uid, "message": "I only want Pune, min 6 LPA"})
    prof = client.get(f"/api/users/{uid}").json()["profile"]
    assert prof["locations"] == ["Pune"] and prof["min_salary_lpa"] == 6
    history = client.fake_llm.chat_calls[-1]
    assert any(m.get("content") == "find me backend internships" for m in history)  # short-term memory

    # digest: hallucinated ids twice -> safe fallback; valid ids -> LLM picks
    bad = {"summary": "x", "picks": [{"job_id": 999999, "fit_score": 90, "why": ["?"], "missing_skills": []}] * 3}
    client.fake_llm.structured_replies += [bad, bad]
    d = client.get(f"/api/users/{uid}/digest").json()
    assert d["meta"]["source"] == "fallback" and len(d["picks"]) == 3

    client.fake_llm.structured_replies.append(pick_first_three)
    d = client.get(f"/api/users/{uid}/digest").json()
    assert d["meta"]["source"] == "llm" and d["meta"]["first_try_valid"] and len(d["picks"]) == 3

    s = client.get("/api/stats").json()
    assert s["users"] == 1 and s["searches"] == 2 and s["chats"] == 2 and s["duplicates_merged"] == 4
    assert client.get("/").status_code == 200
