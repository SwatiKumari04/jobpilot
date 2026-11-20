"""LLM + embedding wrappers, through the real OpenAI SDK with the HTTP layer faked."""
import json

from fakes import completion, fake_openai

from jobpilot.agent import TOOLS
from jobpilot.config import get_settings
from jobpilot.llm import OpenAICompatibleEmbedder, OpenAICompatibleLLM
from jobpilot.schemas import CandidateProfile

SETTINGS = get_settings()
PROFILE = {"headline": "EE student", "skills": ["python", "sql"], "target_roles": ["Backend Intern"],
           "years_experience": 0, "seniority": "intern", "locations": [], "education": "B.Tech"}


# ------------------------------------------------------------------ LLM wrapper
def test_chat_keeps_gemini_thought_signature():
    call = {"id": "c1", "type": "function", "function": {"name": "search_jobs", "arguments": "{\"query\":\"x\"}"},
            "extra_content": {"google": {"thought_signature": "SIG123"}}}
    client, _ = fake_openai(lambda body, path: (200, completion(None, [call])))
    msg = OpenAICompatibleLLM(SETTINGS, client=client).chat([{"role": "user", "content": "hi"}], tools=TOOLS)
    assert msg["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "SIG123"
    assert "content" not in msg  # Gemini rejects empty content


def test_structured_native_path():
    client, reqs = fake_openai(lambda body, path: (200, completion(json.dumps(PROFILE))))
    obj, stats = OpenAICompatibleLLM(SETTINGS, client=client).structured([{"role": "user", "content": "cv"}], CandidateProfile)
    assert obj.seniority == "intern" and stats == {"path": "native", "attempts": 1, "first_try_valid": True}
    assert reqs[0]["body"]["response_format"]["type"] == "json_schema"


def test_structured_falls_back_and_repairs_invalid_json():
    replies = iter([
        (400, {"error": {"message": "response_format not supported", "type": "invalid_request_error"}}),
        (200, completion("```json\n{\"headline\": \"x\"}\n```")),           # missing fields -> invalid
        (200, completion("Sure! " + json.dumps(PROFILE))),                 # repaired
    ])
    client, reqs = fake_openai(lambda body, path: next(replies))
    obj, stats = OpenAICompatibleLLM(SETTINGS, client=client).structured([{"role": "user", "content": "cv"}], CandidateProfile)
    assert obj.skills == ["python", "sql"]
    assert stats["path"] == "json_fallback" and stats["attempts"] == 3
    assert "did not validate" in reqs[-1]["body"]["messages"][-1]["content"]


def test_embedder_truncates_when_dimensions_unsupported():
    calls = []

    def handler(body, path):
        calls.append(body)
        if "dimensions" in body:
            return 400, {"error": {"message": "dimensions not supported"}}
        data = [{"object": "embedding", "index": i, "embedding": [1.0] * 10} for i in range(len(body["input"]))]
        return 200, {"object": "list", "data": data, "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1}}

    client, _ = fake_openai(handler)
    emb = OpenAICompatibleEmbedder(SETTINGS, client=client)
    emb.dim = 4
    vecs = emb.embed(["a", "b"])
    assert len(vecs) == 2 and len(vecs[0]) == 4 and abs(vecs[0][0] - 0.5) < 1e-9
    assert "dimensions" not in calls[-1]


def test_embedder_waits_and_retries_on_per_minute_rate_limit():
    calls = []

    def handler(body, path):
        calls.append(body)
        if len(calls) <= 2:
            return 429, {"error": {"message": "Quota exceeded per minute. Please retry in 0.01s.", "code": 429}}
        data = [{"object": "embedding", "index": None, "embedding": [1.0] * 8} for _ in body["input"]]
        return 200, {"object": "list", "data": data, "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1}}

    client, _ = fake_openai(handler)
    emb = OpenAICompatibleEmbedder(SETTINGS, client=client)
    emb.dim = 4
    assert len(emb.embed(["a", "b"])) == 2 and len(calls) == 3


def test_embedder_gives_up_on_daily_quota():
    import openai
    import pytest

    client, reqs = fake_openai(lambda b, p: (429, {"error": {"message": "Quota exceeded per day.", "code": 429}}))
    emb = OpenAICompatibleEmbedder(SETTINGS, client=client)
    with pytest.raises(openai.RateLimitError):
        emb.embed(["a"])
    assert len(reqs) == 1
