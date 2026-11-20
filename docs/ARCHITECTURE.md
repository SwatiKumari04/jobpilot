# Architecture

## Layered design

Every request flows through the same layers, and each layer has one responsibility:

```
api/routes/    which URL maps to which handler; validate input, call a service, shape the response
api/deps.py    cross-cutting pieces: settings, LLM/embedding clients, rate limiting, user lookup
services/      business logic: profiles (resume -> profile), chat, daily digest, feedback
agent/         the tool-calling loop, the tools, and short-term memory
search/        hybrid retrieval (keyword + vector -> RRF) and personalised ranking
ingest/        the jobs pipeline (dedupe, skills, ghost score, re-embed only what changed) + sources
llm/           thin clients for the chat model and the embedding model, plus versioned prompts
db/            connection pool + schema.sql (Postgres + pgvector)
schemas/       Pydantic models: LLM structured outputs and API request bodies
utils/         pure helpers (skills vocabulary, text cleanup, hashing) with no DB or network
```

Rules I follow to keep the codebase predictable:

- Only `config.py` reads environment variables.
- Routes contain no business logic. They call `services/`, `search/` or `ingest/`.
- `utils/` and `search/ranking.py` are pure functions, so they are unit-tested without a database.
- The LLM is called only through `llm/`. Swapping Gemini for OpenAI, Groq or a local model means changing `LLM_BASE_URL` and `LLM_MODEL` in `.env`, with no code changes.
- Every LLM output is validated (Pydantic) and grounded (job ids must come from our own index) before a user sees it. If validation fails, the system falls back to a non-LLM path instead of failing.

## Data model

| Table | What it stores |
|---|---|
| `jobs` | Every listing: text, skills, level, salary, `embedding vector(768)` (HNSW index), `tsv` full-text column (GIN index), `content_hash`, `dedupe_key`, `duplicate_of`, `ghost_score`, `is_active` |
| `users` | Resume text, `profile` (LLM structured output), `prefs` (long-term memory the agent writes), `profile_embedding` |
| `feedback` | Likes (+1) and dislikes (-1) per user and job, used for personalisation |
| `messages` | Chat history, used as the agent's short-term memory |
| `events` | Product analytics (signups, searches, chats, digests) |

## Flow 1: ingestion (nightly)

```
Remotive API -> normalise -> extract skills (alias dictionary, no LLM)
  -> already seen and content_hash unchanged?  -> just mark it as seen
  -> already seen, content changed?            -> update the row and clear the embedding
  -> new, but dedupe_key matches a live job?   -> cross-post: bump repost_count on the original
  -> brand new                                 -> insert it
-> embed only the rows without an embedding -> expire listings missing from the snapshot
-> recompute ghost scores (listings age every day)
```

Calling an LLM for every listing would not scale to millions of jobs, so ingestion uses cheap, deterministic steps. The LLM is used per user, not per job.

## Flow 2: search and ranking

```
query -> [keyword: Postgres full-text, top 50]  +  [vector: pgvector cosine, top 50]
      -> Reciprocal Rank Fusion: score = sum of 1 / (60 + rank)
      -> personalised re-rank: 0.40 relevance + 0.20 profile similarity + 0.20 skill overlap
         + 0.08 location + 0.07 experience + 0.05 recency + 0.10 taste (likes - dislikes)
         - 0.30 ghost score - salary-floor penalty
      -> reasons ("Matches 4/6 skills", "Remote", ...) and missing skills for each job
```

RRF combines the two lists by rank alone, so a full-text score never has to be compared with a cosine similarity. Query embeddings sit in an LRU cache, so a repeated search makes no API call. If the embedding API is down, search falls back to keyword-only results.

## Flow 3: the agent

```
system prompt (+ the user's profile = long-term memory) + last 12 turns (short-term memory) + new message
  -> LLM returns tool_calls?  -> run each tool -> append the results -> ask the LLM again (max 6 steps)
  -> LLM returns text         -> done; check that every [#id] it cites came from a tool result
```

Tools: `search_jobs`, `get_job_details`, `update_preferences`, `save_job`, `get_saved_jobs`. Bad tool arguments are returned to the model as an error message, so it can correct its own call.

Gemini 3 detail: tool calls carry `extra_content.google.thought_signature`, which must be sent back unchanged on the next request or the API returns a 400 error. `llm/chat.py` keeps it, and `tests/test_agent.py` checks that it round-trips.

## Flow 4: daily digest (top 3 jobs)

```
profile -> search (top 10 candidates) -> LLM picks 3 as structured output (Digest schema)
  -> validate: exactly 3, unique, every job_id is in the candidate list
  -> invalid? one repair round with the errors -> still invalid? fall back to the ranking's top 3
```

## Evaluation

- `evals/retrieval_eval.py` measures P@10, MRR and nDCG@10 for keyword, vector and hybrid search on labelled synthetic jobs, with a separate score for paraphrased queries.
- `evals/llm_eval.py` measures skill F1 and seniority accuracy for resume parsing, first-try schema validity, the grounding rate, tool-selection accuracy and whether preferences are remembered.
- Prompts are versioned in `llm/prompts.py`, and the version is saved with every eval result, so a prompt change can be compared before and after.
