"""The JobPilot agent: an LLM that calls tools in a loop and remembers context.

Memory:
  * short-term = the last N chat turns (messages table)
  * long-term  = the user's profile + preferences (users table), which the
                 agent itself updates through the `update_preferences` tool.

The loop (a.k.a. ReAct / function calling):
  1. send conversation + tool definitions to the LLM
  2. if it asks for tool calls -> run them, append results, go to 1
  3. if it answers in text -> done
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

from ..llm import ChatLLM, prompts
from .tools import TOOLS, ToolBox

log = logging.getLogger(__name__)
MAX_STEPS = 6


@dataclass
class AgentResult:
    reply: str
    jobs: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    steps: int = 0
    ungrounded_ids: list[int] = field(default_factory=list)


CITATION = re.compile(r"\[#(\d+)\]")


def run_agent(llm: ChatLLM, toolbox: ToolBox, message: str, history: list[dict], max_steps: int = MAX_STEPS) -> AgentResult:
    profile_json = json.dumps(toolbox.user.get("effective_profile") or {}, ensure_ascii=False)
    messages: list[dict] = [
        {"role": "system", "content": prompts.AGENT_SYSTEM.format(profile=profile_json)},
        *history,
        {"role": "user", "content": message},
    ]
    trace: list[dict] = []
    reply = ""
    steps = 0
    for steps in range(1, max_steps + 1):
        assistant = llm.chat(messages, tools=TOOLS)
        messages.append(assistant)  # unchanged, so provider extras (thought signatures) round-trip
        calls = assistant.get("tool_calls") or []
        if not calls:
            reply = (assistant.get("content") or "").strip()
            break
        for call in calls:
            fn = call.get("function") or {}
            name, raw = fn.get("name", ""), fn.get("arguments") or "{}"
            t0 = time.perf_counter()
            result = toolbox.run(name, raw)
            trace.append({"tool": name, "args": raw[:300], "ms": round((time.perf_counter() - t0) * 1000),
                          "ok": "error" not in result})
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": json.dumps(result, default=str, ensure_ascii=False)})
    else:
        reply = "I searched a few times but couldn't finish my answer. Here are the most relevant jobs I found."

    if not reply:
        reply = "Here is what I found."
    cited = [int(x) for x in CITATION.findall(reply)]
    ungrounded = [i for i in cited if i not in toolbox.shown]
    if ungrounded:
        log.warning("agent cited job ids that no tool returned: %s", ungrounded)
    order = cited + [i for i in toolbox.shown if i not in cited]
    jobs = [toolbox.shown[i] for i in order if i in toolbox.shown][:6]
    return AgentResult(reply=reply, jobs=jobs, trace=trace, steps=steps, ungrounded_ids=ungrounded)
