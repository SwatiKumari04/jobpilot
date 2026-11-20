"""Thin wrappers around the LLM + embedding APIs.

Why no LangChain? The whole agent is ~100 lines when you call the API
directly, and you can see (and debug) every request. We use the official
OpenAI Python SDK pointed at Gemini's OpenAI-compatible endpoint, so the
same code also works with OpenAI, Groq, OpenRouter, a local vLLM, etc.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ..config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    pass


class ChatLLM(Protocol):
    model: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict: ...

    def structured(self, messages: list[dict], schema: type[T], max_repairs: int = 2) -> tuple[T, dict]: ...


def _extract_json(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences or add a sentence. Pull out the object."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


class OpenAICompatibleLLM:
    def __init__(self, settings: Settings, client: OpenAI | None = None):
        self.model = settings.llm_model
        self.client = client or OpenAI(
            api_key=settings.llm_api_key or "missing-key",
            base_url=settings.llm_base_url,
            timeout=60.0,
            max_retries=3,  # SDK retries 429 / 5xx with exponential backoff
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self.client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.model_dump(exclude_none=True)
        # Keep only fields that are valid to send back, PLUS provider extras:
        # Gemini 3 puts a required `extra_content.google.thought_signature` on each
        # tool call; it MUST be sent back unchanged or the next request fails (400).
        out: dict[str, Any] = {"role": "assistant"}
        if raw.get("content"):  # Gemini rejects empty-string content
            out["content"] = raw["content"]
        if raw.get("tool_calls"):
            out["tool_calls"] = [
                {k: tc[k] for k in ("id", "type", "function", "extra_content") if k in tc}
                for tc in raw["tool_calls"]
            ]
        if raw.get("extra_content"):
            out["extra_content"] = raw["extra_content"]
        return out

    def structured(self, messages: list[dict], schema: type[T], max_repairs: int = 2) -> tuple[T, dict]:
        """Return (validated pydantic object, stats).

        1) Try native structured output (JSON schema enforced by the provider).
        2) If that is unsupported or fails validation, fall back to "JSON in the
           prompt" + Pydantic validation, feeding the validation error back to
           the model so it can repair its own output (up to max_repairs times).
        """
        stats = {"path": "native", "attempts": 0, "first_try_valid": False}
        try:
            stats["attempts"] += 1
            completion = self.client.chat.completions.parse(
                model=self.model, messages=messages, response_format=schema
            )
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                stats["first_try_valid"] = True
                return parsed, stats
        except (openai.BadRequestError, openai.UnprocessableEntityError, ValidationError,
                json.JSONDecodeError, openai.LengthFinishReasonError, openai.ContentFilterFinishReasonError) as exc:
            log.warning("native structured output failed (%s); using JSON fallback", type(exc).__name__)

        stats["path"] = "json_fallback"
        schema_json = json.dumps(schema.model_json_schema())
        convo = [
            {
                "role": "system",
                "content": "Reply with ONLY one JSON object (no markdown, no prose) that matches this JSON Schema:\n"
                + schema_json,
            },
            *messages,
        ]
        last_error = ""
        for attempt in range(max_repairs + 1):
            stats["attempts"] += 1
            resp = self.client.chat.completions.create(model=self.model, messages=convo)
            text = resp.choices[0].message.content or ""
            try:
                return schema.model_validate_json(_extract_json(text)), stats
            except ValidationError as exc:
                last_error = str(exc)[:1500]
                convo = convo + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That JSON did not validate:\n{last_error}\nReturn the corrected JSON object only."},
                ]
        raise StructuredOutputError(f"could not get valid {schema.__name__}: {last_error}")
