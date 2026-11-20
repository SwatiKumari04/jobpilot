"""Skill vocabulary: extract and normalise technical skills (no LLM needed)."""
from __future__ import annotations

import re

# Canonical skill -> aliases that should map to it. Used to extract skills from
# job text cheaply at ingest time (calling an LLM per listing does not scale to
# millions of jobs) and to normalise skills parsed from resumes.
SKILL_ALIASES: dict[str, list[str]] = {
    "python": ["python", "python3"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "c": ["c programming", "embedded c"],
    "go": ["golang", "go lang"],
    "rust": ["rust"],
    "javascript": ["javascript", "js", "es6"],
    "typescript": ["typescript", "ts"],
    "kotlin": ["kotlin"],
    "swift": ["swift"],
    "sql": ["sql", "t-sql", "pl/sql"],
    "postgresql": ["postgresql", "postgres", "psql"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "fastapi": ["fastapi"],
    "django": ["django"],
    "flask": ["flask"],
    "node.js": ["node.js", "nodejs", "node"],
    "express": ["express.js", "expressjs", "express"],
    "spring boot": ["spring boot", "springboot", "spring"],
    "react": ["react", "react.js", "reactjs"],
    "next.js": ["next.js", "nextjs"],
    "angular": ["angular"],
    "vue": ["vue", "vue.js"],
    "html": ["html", "html5"],
    "css": ["css", "css3", "tailwind", "sass"],
    "rest api": ["rest api", "rest apis", "restful", "rest"],
    "graphql": ["graphql"],
    "grpc": ["grpc"],
    "websocket": ["websocket", "websockets"],
    "docker": ["docker", "containers", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "gcp": ["gcp", "google cloud"],
    "azure": ["azure"],
    "terraform": ["terraform"],
    "ci/cd": ["ci/cd", "cicd", "github actions", "jenkins", "gitlab ci"],
    "linux": ["linux", "unix", "bash", "shell scripting"],
    "git": ["git", "github", "gitlab"],
    "microservices": ["microservices", "microservice"],
    "kafka": ["kafka"],
    "spark": ["spark", "pyspark"],
    "airflow": ["airflow"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "keras"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"],
    "computer vision": ["computer vision", "opencv"],
    "llm": ["llm", "llms", "large language model", "large language models", "gpt", "gemini", "claude"],
    "rag": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "embeddings": ["embeddings", "embedding", "vector search", "semantic search"],
    "vector database": ["pgvector", "pinecone", "weaviate", "qdrant", "milvus", "vector database", "vector db"],
    "prompt engineering": ["prompt engineering", "prompting"],
    "agents": ["ai agents", "llm agents", "agentic", "tool calling", "function calling"],
    "langchain": ["langchain", "langgraph", "llamaindex"],
    "data analysis": ["data analysis", "analytics"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "excel": ["excel"],
    "statistics": ["statistics", "a/b testing"],
    "android": ["android"],
    "ios": ["ios"],
    "flutter": ["flutter", "dart"],
    "react native": ["react native"],
    "selenium": ["selenium"],
    "testing": ["unit testing", "pytest", "jest", "test automation", "qa"],
    "system design": ["system design", "distributed systems"],
    "dsa": ["data structures", "algorithms", "dsa"],
    "embedded": ["embedded systems", "microcontroller", "rtos", "firmware"],
    "figma": ["figma"],
}

_ALIAS_TO_SKILL: dict[str, str] = {}
for canonical, aliases in SKILL_ALIASES.items():
    for a in aliases + [canonical]:
        _ALIAS_TO_SKILL[a.lower()] = canonical

# Longest aliases first so "react native" wins over "react".
_ALIAS_PATTERN = re.compile(
    r"(?<![a-z0-9])("
    + "|".join(re.escape(a) for a in sorted(_ALIAS_TO_SKILL, key=len, reverse=True))
    + r")(?![a-z0-9+#])",
    re.I,
)

_AMBIGUOUS = {"go", "c", "rest", "node", "spring", "express", "ts", "js", "ml", "qa", "rust", "swift", "excel", "claude", "gemini"}


def extract_skills(text: str) -> list[str]:
    """Return canonical skills mentioned in text (order of first appearance)."""
    found: list[str] = []
    for m in _ALIAS_PATTERN.finditer(text or ""):
        alias = m.group(1).lower()
        if alias in _AMBIGUOUS and not _looks_technical(text, m.start()):
            continue
        skill = _ALIAS_TO_SKILL[alias]
        if skill not in found:
            found.append(skill)
    return found


def _looks_technical(text: str, pos: int) -> bool:
    # Ambiguous words ("go", "rest", "swift") only count inside a list of tech words.
    window = text[max(0, pos - 40) : pos + 40]
    return window.count(",") >= 1 or "/" in window or "|" in window


def normalize_skill(skill: str) -> str:
    s = skill.strip().lower()
    return _ALIAS_TO_SKILL.get(s, s)


def normalize_skills(skills: list[str]) -> list[str]:
    out: list[str] = []
    for s in skills:
        n = normalize_skill(s)
        if n and n not in out:
            out.append(n)
    return out
