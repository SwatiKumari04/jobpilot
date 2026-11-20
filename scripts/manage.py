"""Command-line tasks.

    python -m scripts.manage init-db              # create tables + indexes
    python -m scripts.manage seed --n 300         # synthetic demo jobs (used by evals)
    python -m scripts.manage ingest-remotive      # real remote jobs from Remotive
    python -m scripts.manage nightly              # what the nightly GitHub Action runs
    python -m scripts.manage stats
"""
from __future__ import annotations

import argparse
import json
import logging
import time

from jobpilot import db, ingest
from jobpilot.config import get_settings
from jobpilot.llm import get_embedder
from jobpilot.ingest.sources import remotive, synthetic


def main() -> None:
    parser = argparse.ArgumentParser(prog="manage")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    seed = sub.add_parser("seed")
    seed.add_argument("--n", type=int, default=300)
    rem = sub.add_parser("ingest-remotive")
    rem.add_argument("--limit", type=int, default=150, help="max jobs per category")
    sub.add_parser("nightly")
    sub.add_parser("stats")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    db.init_pool(settings)
    db.init_schema(settings)
    embedder = get_embedder(settings)
    t0 = time.perf_counter()

    if args.cmd == "init-db":
        print("schema ready")
    elif args.cmd == "seed":
        jobs = synthetic.generate(args.n)
        print(json.dumps(ingest.ingest_jobs(jobs, embedder).model_dump(), indent=2))
    elif args.cmd == "ingest-remotive":
        jobs = remotive.fetch(limit_per_category=args.limit)
        print(f"fetched {len(jobs)} jobs from Remotive")
        print(json.dumps(ingest.ingest_jobs(jobs, embedder, expire_missing_for_source="remotive").model_dump(), indent=2))
    elif args.cmd == "nightly":
        jobs = remotive.fetch()
        stats = ingest.ingest_jobs(jobs, embedder, expire_missing_for_source="remotive")
        stats.expired += ingest.expire_old(90)
        rescored = ingest.refresh_ghost_scores()
        print(json.dumps({**stats.model_dump(), "ghost_rescored": rescored}, indent=2))
    elif args.cmd == "stats":
        with db.conn() as c:
            print(json.dumps(c.execute(
                "SELECT source, count(*) FILTER (WHERE is_active AND duplicate_of IS NULL) AS active, "
                "count(*) FILTER (WHERE duplicate_of IS NOT NULL) AS duplicates, "
                "count(*) FILTER (WHERE ghost_score >= 0.7) AS ghost, count(*) AS total "
                "FROM jobs GROUP BY source ORDER BY source").fetchall(), indent=2, default=str))
    print(f"done in {time.perf_counter() - t0:.1f}s")
    db.close_pool()


if __name__ == "__main__":
    main()
