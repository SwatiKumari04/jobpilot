"""Retrieval eval: does hybrid search beat keyword-only and vector-only?

Uses the synthetic jobs (seed them first: `python -m scripts.manage seed`),
whose `category` gives ground-truth relevance. Half the queries are
paraphrases that share few words with the listings — that is where keyword
search fails and embeddings should help.

    python -m evals.retrieval_eval
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from jobpilot import db
from jobpilot.config import get_settings
from jobpilot.llm import get_embedder
from jobpilot.search import Filters, search

from .metrics import ndcg_at_k, precision_at_k, reciprocal_rank

QUERIES: list[tuple[str, str]] = [
    # (query, relevant category) — keyword-style
    ("python backend developer fastapi postgresql", "backend"),
    ("java spring boot microservices engineer", "backend"),
    ("react typescript frontend developer", "frontend"),
    ("MERN full stack developer node express mongodb", "fullstack"),
    ("LLM engineer RAG embeddings", "ml_ai"),
    ("machine learning engineer pytorch", "ml_ai"),
    ("data analyst sql power bi", "data"),
    ("devops kubernetes terraform aws", "devops"),
    ("android kotlin developer", "mobile"),
    ("selenium test automation engineer", "qa"),
    ("embedded firmware microcontroller c", "embedded"),
    # paraphrases — few shared words with the listings
    ("someone to create the server side of our app and its database", "backend"),
    ("make websites look beautiful and feel fast for users", "frontend"),
    ("one person who can handle both the screens and the server code", "fullstack"),
    ("build chatbots that understand language and answer from documents", "ml_ai"),
    ("turn numbers into charts and business insights for managers", "data"),
    ("keep production servers alive and automate releases", "devops"),
    ("create apps for smartphones", "mobile"),
    ("catch software defects before release", "qa"),
    ("program chips on circuit boards close to the hardware", "embedded"),
]
K = 10


def main() -> None:
    settings = get_settings()
    db.init_pool(settings)
    embedder = get_embedder(settings)
    with db.conn() as c:
        counts = {r["category"]: r["n"] for r in c.execute(
            "SELECT category, count(*) AS n FROM jobs WHERE source = 'synthetic' AND is_active "
            "AND duplicate_of IS NULL AND ghost_score < 0.7 GROUP BY category").fetchall()}
    if not counts:
        raise SystemExit("No synthetic jobs found. Run: python -m scripts.manage seed")

    results: dict[str, dict] = {}
    per_query: list[dict] = []
    for mode in ("keyword", "vector", "hybrid"):
        p, mrr, ndcg, lat = [], [], [], []
        for query, cat in QUERIES:
            t0 = time.perf_counter()
            rows = search(query, embedder, filters=Filters(sources=["synthetic"]), limit=K, mode=mode, personalize=False)
            lat.append((time.perf_counter() - t0) * 1000)
            ranked = [r["category"] == cat for r in rows]
            ranked += [False] * (K - len(ranked))
            p.append(precision_at_k(ranked, K))
            mrr.append(reciprocal_rank(ranked))
            ndcg.append(ndcg_at_k(ranked, K, counts.get(cat, 0)))
            per_query.append({"mode": mode, "query": query, "category": cat, "p@10": p[-1], "ndcg@10": round(ndcg[-1], 3)})
        lat.sort()
        results[mode] = {
            "precision@10": round(sum(p) / len(p), 3),
            "mrr": round(sum(mrr) / len(mrr), 3),
            "ndcg@10": round(sum(ndcg) / len(ndcg), 3),
            "paraphrase_ndcg@10": round(sum(ndcg[11:]) / len(ndcg[11:]), 3),
            "p50_ms": round(lat[len(lat) // 2], 1),
        }

    print(f"\nRetrieval eval — {len(QUERIES)} queries, {sum(counts.values())} labelled jobs, embed model={settings.embed_model}\n")
    print("| mode    | P@10  | MRR   | nDCG@10 | nDCG@10 (paraphrases) | p50 ms |")
    print("|---------|-------|-------|---------|-----------------------|--------|")
    for mode, m in results.items():
        print(f"| {mode:<7} | {m['precision@10']:.3f} | {m['mrr']:.3f} | {m['ndcg@10']:.3f}   | "
              f"{m['paraphrase_ndcg@10']:.3f}                 | {m['p50_ms']:>6} |")
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "retrieval.json").write_text(json.dumps({"summary": results, "per_query": per_query}, indent=2))
    print(f"\nSaved {out / 'retrieval.json'}")
    db.close_pool()


if __name__ == "__main__":
    main()
