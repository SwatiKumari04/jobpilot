"""Pydantic schemas: LLM structured outputs and API request bodies."""
from .llm_outputs import CandidateProfile, Digest, DigestPick, Seniority

__all__ = ["CandidateProfile", "Digest", "DigestPick", "Seniority"]
