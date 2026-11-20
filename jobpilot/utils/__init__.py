"""Pure helper functions (no DB, no network): easy to unit-test."""
from .skills import SKILL_ALIASES, extract_skills, normalize_skill, normalize_skills
from .text import content_hash, dedupe_key, ghost_reasons, ghost_score, strip_html

__all__ = ["SKILL_ALIASES", "extract_skills", "normalize_skill", "normalize_skills",
           "content_hash", "dedupe_key", "ghost_reasons", "ghost_score", "strip_html"]
