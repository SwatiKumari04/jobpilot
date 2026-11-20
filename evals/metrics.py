"""Standard IR metrics with binary relevance."""
from __future__ import annotations

import math


def precision_at_k(ranked: list[bool], k: int) -> float:
    top = ranked[:k]
    return sum(top) / k if k else 0.0


def reciprocal_rank(ranked: list[bool]) -> float:
    for i, rel in enumerate(ranked, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[bool], k: int, total_relevant: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, rel in enumerate(ranked[:k], start=1) if rel)
    ideal_hits = min(k, total_relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def f1(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    tp = len(predicted & expected)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(expected) if expected else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)
