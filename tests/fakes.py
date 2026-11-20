"""Test doubles: a scripted LLM (no network) and helpers to fake OpenAI HTTP responses."""
from __future__ import annotations

import json
from typing import Callable

import httpx
from openai import OpenAI
from pydantic import BaseModel


class ScriptedLLM:
    """Returns pre-written assistant messages in order; records every request."""

    model = "scripted"

    def __init__(self, chat_replies: list[dict] | None = None, structured_replies: list | None = None):
        self.chat_replies = list(chat_replies or [])
        self.structured_replies = list(structured_replies or [])
        self.chat_calls: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.chat_calls.append([dict(m) for m in messages])
        if not self.chat_replies:
            return {"role": "assistant", "content": "Done."}
        return self.chat_replies.pop(0)

    def structured(self, messages, schema, max_repairs=2):
        if not self.structured_replies:
            raise RuntimeError("no scripted structured reply")
        reply = self.structured_replies.pop(0)
        if callable(reply):
            reply = reply(messages, schema)
        if isinstance(reply, Exception):
            raise reply
        obj = reply if isinstance(reply, BaseModel) else schema.model_validate(reply)
        return obj, {"path": "scripted", "attempts": 1, "first_try_valid": True}


def tool_call(name: str, args: dict, call_id: str = "call_1", signature: str | None = None) -> dict:
    call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    if signature:
        call["extra_content"] = {"google": {"thought_signature": signature}}
    return {"role": "assistant", "tool_calls": [call]}


def completion(content: str | None = None, tool_calls: list | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "fake",
        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def fake_openai(handler: Callable[[dict, str], tuple[int, dict]]) -> tuple[OpenAI, list[dict]]:
    """OpenAI SDK client whose HTTP layer calls handler(json_body, path) -> (status, json)."""
    requests: list[dict] = []

    def transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        requests.append({"path": request.url.path, "body": body})
        status, payload = handler(body, request.url.path)
        return httpx.Response(status, json=payload)

    client = OpenAI(api_key="test", base_url="http://fake.local/v1", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(transport)))
    return client, requests
