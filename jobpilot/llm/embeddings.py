"""Embedding clients: any OpenAI-compatible API (Gemini by default) + an offline hash embedder."""
from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from typing import Any, Protocol

import openai
from openai import OpenAI

from ..config import Settings

log = logging.getLogger(__name__)
_RETRY_IN = re.compile(r"retry in ([0-9.]+)\s*s", re.I)


def _wait_seconds(exc: Exception, default: float = 30.0) -> float:
    """Gemini's 429 message says 'Please retry in 15.3s'; use that if present."""
    m = _RETRY_IN.search(str(exc))
    return min(90.0, float(m.group(1)) + 2.0) if m else default


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]: ...


def fit_dim(vec: list[float], dim: int) -> list[float]:
    """Matryoshka (MRL) embeddings can be truncated; re-normalise to unit length."""
    v = list(vec[:dim])
    if len(v) < dim:
        v += [0.0] * (dim - len(v))
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class OpenAICompatibleEmbedder:
    def __init__(self, settings: Settings, batch_size: int = 50, client: OpenAI | None = None):
        self.dim = settings.embed_dim
        self.model = settings.embed_model
        self.batch_size = batch_size
        self.client = client or OpenAI(
            api_key=settings.embed_api_key or "missing-key",
            base_url=settings.embed_base_url,
            timeout=60.0,
            max_retries=1,  # our own _create() handles per-minute limits with a longer wait
        )
        self._send_dimensions = True
        self.max_rate_limit_waits = 8

    def _prefix(self, text: str, kind: str) -> str:
        # Gemini Embedding 2 is trained with task prefixes (asymmetric retrieval).
        if "gemini-embedding-2" not in self.model:
            return text
        if kind == "query":
            return f"task: search result | query: {text}"
        return f"title: none | text: {text}"

    def _create(self, kwargs: dict[str, Any]):
        """Call the API; on a free-tier per-minute limit (429), wait and try again."""
        for attempt in range(self.max_rate_limit_waits + 1):
            try:
                return self.client.embeddings.create(**kwargs)
            except openai.RateLimitError as exc:
                if attempt == self.max_rate_limit_waits or "per day" in str(exc).lower():
                    raise
                wait = _wait_seconds(exc)
                log.warning("embedding rate limit hit; waiting %.0fs then retrying (%d/%d)",
                            wait, attempt + 1, self.max_rate_limit_waits)
                time.sleep(wait)

    def _call(self, batch: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": batch}
        if self._send_dimensions:
            kwargs["dimensions"] = self.dim
        try:
            resp = self._create(kwargs)
        except openai.BadRequestError:
            if not self._send_dimensions:
                raise
            # Some providers reject `dimensions`; we can truncate ourselves (MRL).
            self._send_dimensions = False
            kwargs.pop("dimensions")
            resp = self._create(kwargs)
        data = list(resp.data)
        # Gemini's OpenAI-compatible endpoint may leave `index` empty; then keep its order.
        if all(d.index is not None for d in data):
            data.sort(key=lambda d: d.index)
        return [d.embedding for d in data]

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        out: list[list[float]] = []
        prepared = [self._prefix(t[:6000], kind) for t in texts]
        for i in range(0, len(prepared), self.batch_size):
            batch = prepared[i : i + self.batch_size]
            vectors = self._call(batch)
            if len(vectors) != len(batch):  # provider merged the batch: go one by one
                vectors = [self._call([t])[0] for t in batch]
            out.extend(fit_dim(v, self.dim) for v in vectors)
            if i + self.batch_size < len(prepared):
                time.sleep(0.5)  # be gentle with free-tier rate limits
        return out


class HashEmbedder:
    """Offline embedder (feature hashing of words + bigrams). Not semantic, but
    deterministic and free — used in tests and when no API key is available."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dim
            words = re.findall(r"[a-z0-9+#]+", text.lower())
            grams = words + [a + "_" + b for a, b in zip(words, words[1:])]
            for g in grams:
                h = int(hashlib.md5(g.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
            out.append(fit_dim(v, self.dim))
        return out
