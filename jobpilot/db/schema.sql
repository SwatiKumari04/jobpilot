-- JobPilot schema. Safe to run many times (everything is IF NOT EXISTS).
-- __EMBED_DIM__ is replaced with settings.embed_dim (default 768) at startup.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- jobs index
CREATE TABLE IF NOT EXISTS jobs (
    id              BIGSERIAL PRIMARY KEY,
    source          TEXT NOT NULL,              -- 'remotive', 'synthetic', ...
    external_id     TEXT NOT NULL,              -- id inside that source
    dedupe_key      TEXT NOT NULL,              -- same job posted on many boards -> same key
    title           TEXT NOT NULL,
    company         TEXT NOT NULL,
    location        TEXT NOT NULL DEFAULT '',
    remote          BOOLEAN NOT NULL DEFAULT FALSE,
    category        TEXT NOT NULL DEFAULT '',
    level           TEXT NOT NULL DEFAULT '',    -- intern / fresher / junior / mid / senior
    experience_min  REAL NOT NULL DEFAULT 0,
    experience_max  REAL NOT NULL DEFAULT 30,
    salary_min_lpa  REAL,
    salary_max_lpa  REAL,
    skills          TEXT[] NOT NULL DEFAULT '{}',
    description     TEXT NOT NULL DEFAULT '',
    url             TEXT NOT NULL DEFAULT '',
    posted_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    repost_count    INT NOT NULL DEFAULT 0,
    duplicate_of    BIGINT REFERENCES jobs(id) ON DELETE SET NULL, -- set when this is a cross-post of another job
    ghost_score     REAL NOT NULL DEFAULT 0,      -- 0..1, high = probably a fake / stale listing
    content_hash    TEXT NOT NULL,                -- re-embed only when content changes
    embedding       vector(__EMBED_DIM__),
    tsv             tsvector GENERATED ALWAYS AS (
                        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                        setweight(to_tsvector('english', coalesce(company, '')), 'B') ||
                        setweight(to_tsvector('english', coalesce(description, '')), 'C')
                    ) STORED,
    UNIQUE (source, external_id)
);

-- Approximate-nearest-neighbour index for semantic search (cosine distance).
CREATE INDEX IF NOT EXISTS jobs_embedding_hnsw ON jobs USING hnsw (embedding vector_cosine_ops);
-- Full-text index for keyword search.
CREATE INDEX IF NOT EXISTS jobs_tsv_gin ON jobs USING gin (tsv);
CREATE INDEX IF NOT EXISTS jobs_dedupe_key ON jobs (dedupe_key);
CREATE INDEX IF NOT EXISTS jobs_active_posted ON jobs (is_active, posted_at DESC);

-- ---------------------------------------------------------------- users
CREATE TABLE IF NOT EXISTS users (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name               TEXT NOT NULL DEFAULT '',
    resume_text        TEXT NOT NULL DEFAULT '',
    profile            JSONB NOT NULL DEFAULT '{}'::jsonb,  -- structured output from the LLM
    prefs              JSONB NOT NULL DEFAULT '{}'::jsonb,  -- long-term memory the agent updates
    profile_embedding  vector(__EMBED_DIM__),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Likes / dislikes / saves. Used for personalisation.
CREATE TABLE IF NOT EXISTS feedback (
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_id      BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    signal      SMALLINT NOT NULL CHECK (signal IN (-1, 1)),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, job_id)
);

-- Conversation memory for the agent (short-term memory).
CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_user_time ON messages (user_id, created_at DESC);

-- Simple product analytics: proves real people use it ("used by N students").
CREATE TABLE IF NOT EXISTS events (
    id          BIGSERIAL PRIMARY KEY,
    user_id     UUID,
    kind        TEXT NOT NULL,           -- 'signup', 'search', 'chat', 'digest', 'feedback'
    meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS events_kind_time ON events (kind, created_at DESC);
