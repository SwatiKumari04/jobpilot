"""Chat with the agent: load memory -> run the agent loop -> save memory."""
from __future__ import annotations

from ..agent import AgentResult, ToolBox, load_history, run_agent, save_turn
from ..llm import ChatLLM, Embedder


def chat(llm: ChatLLM, embedder: Embedder, user: dict, message: str) -> AgentResult:
    history = load_history(user["id"])
    result = run_agent(llm, ToolBox(user, embedder), message, history)
    save_turn(user["id"], message, result.reply)
    return result
