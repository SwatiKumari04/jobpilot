"""Agent loop (tool calls, grounding, step limit) and digest validation. No network, no DB."""
import json

from fakes import ScriptedLLM, tool_call

from jobpilot.agent import ToolBox, run_agent
from jobpilot.schemas import Digest, DigestPick
from jobpilot.services.digest import validate_digest

PROFILE = {"headline": "EE student", "skills": ["python", "sql"], "target_roles": ["Backend Intern"],
           "years_experience": 0, "seniority": "intern", "locations": [], "education": "B.Tech"}


class FakeToolBox(ToolBox):
    def __init__(self):
        super().__init__({"id": "u1", "effective_profile": PROFILE}, embedder=None)  # type: ignore[arg-type]
        self.calls = []

    def _tool_search_jobs(self, args):
        self.calls.append(("search_jobs", args))
        card = {"id": 7, "title": "Backend Intern", "company": "Nimbus Labs", "location": "Remote"}
        self.shown[7] = card
        return {"count": 1, "jobs": [card]}


def test_agent_runs_tool_then_answers_and_round_trips_signature():
    llm = ScriptedLLM([
        tool_call("search_jobs", {"query": "python backend intern"}, signature="SIG"),
        {"role": "assistant", "content": "Try [#7] Backend Intern at Nimbus Labs."},
    ])
    box = FakeToolBox()
    res = run_agent(llm, box, "find me backend internships", history=[])
    assert box.calls == [("search_jobs", {"query": "python backend intern"})]
    assert res.reply.startswith("Try [#7]") and res.jobs[0]["id"] == 7 and res.ungrounded_ids == []
    second_request = llm.chat_calls[1]
    assert second_request[-2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "SIG"
    assert second_request[-1]["role"] == "tool" and json.loads(second_request[-1]["content"])["count"] == 1


def test_agent_reports_hallucinated_job_ids():
    llm = ScriptedLLM([{"role": "assistant", "content": "Apply to [#999] Dream Job."}])
    res = run_agent(llm, FakeToolBox(), "anything?", history=[])
    assert res.ungrounded_ids == [999]


def test_bad_tool_arguments_are_returned_to_the_model():
    llm = ScriptedLLM([tool_call("search_jobs", {"limit": 3}), {"role": "assistant", "content": "ok"}])
    box = ToolBox({"id": "u1", "effective_profile": PROFILE}, embedder=None)  # type: ignore[arg-type]
    res = run_agent(llm, box, "search", history=[])
    tool_msg = llm.chat_calls[1][-1]
    assert "invalid arguments" in json.loads(tool_msg["content"])["error"]
    assert res.trace[0]["ok"] is False


def test_agent_stops_after_max_steps():
    llm = ScriptedLLM([tool_call("search_jobs", {"query": "x"}, call_id=f"c{i}") for i in range(10)])
    res = run_agent(llm, FakeToolBox(), "loop", history=[], max_steps=3)
    assert res.steps == 3 and "couldn't finish" in res.reply


def test_unknown_tool():
    assert "unknown tool" in FakeToolBox().run("rm_rf", "{}")["error"]


# ------------------------------------------------------------------ digest grounding
def test_digest_validation_catches_hallucinated_and_duplicate_ids():
    pick = lambda i: DigestPick(job_id=i, fit_score=80, why=["fits"], missing_skills=[])
    assert validate_digest(Digest(picks=[pick(1), pick(2), pick(3)], summary="s"), {1, 2, 3, 4}, 3) == []
    problems = validate_digest(Digest(picks=[pick(1), pick(1), pick(99)], summary="s"), {1, 2, 3}, 3)
    assert any("not in the candidate list" in p for p in problems) and "duplicate job ids" in problems
