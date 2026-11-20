"""LLM + embedding clients and the factories the rest of the app uses."""
from __future__ import annotations

from ..config import Settings
from .chat import ChatLLM, OpenAICompatibleLLM, StructuredOutputError
from .embeddings import Embedder, HashEmbedder, OpenAICompatibleEmbedder, fit_dim

__all__ = ["ChatLLM", "Embedder", "HashEmbedder", "OpenAICompatibleEmbedder", "OpenAICompatibleLLM",
           "StructuredOutputError", "fit_dim", "get_embedder", "get_llm", "set_embedder", "set_llm"]

_llm: ChatLLM | None = None
_embedder: Embedder | None = None


def get_llm(settings: Settings) -> ChatLLM:
    global _llm
    if _llm is None:
        _llm = OpenAICompatibleLLM(settings)
    return _llm


def get_embedder(settings: Settings) -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = HashEmbedder(settings.embed_dim) if settings.embed_provider == "hash" else OpenAICompatibleEmbedder(settings)
    return _embedder


def set_llm(llm: ChatLLM | None) -> None:  # used by tests
    global _llm
    _llm = llm


def set_embedder(embedder: Embedder | None) -> None:  # used by tests
    global _embedder
    _embedder = embedder
