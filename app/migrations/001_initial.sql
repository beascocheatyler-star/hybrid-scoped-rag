-- M1: notebooks, sources, chunks, vectors, jobs.
-- Never edit a shipped migration; add a new numbered file instead.

CREATE TABLE app_meta (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE users (
    id            TEXT PRIMARY KEY,
    email         TEXT UNIQUE,
    password_hash TEXT,
    display_name  TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE notebooks (
    id               TEXT PRIMARY KEY,
    owner_id         TEXT NOT NULL REFERENCES users(id),
    title            TEXT NOT NULL,
    description      TEXT NOT NULL DEFAULT '',
    emoji            TEXT NOT NULL DEFAULT '',
    chat_config_json TEXT NOT NULL DEFAULT '{}',
    overview_json    TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE memberships (
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('owner','editor','viewer')),
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (notebook_id, user_id)
);

CREATE TABLE sources (
    id            TEXT PRIMARY KEY,
    notebook_id   TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL CHECK (kind IN ('file','url','youtube','text','audio','image','note')),
    title         TEXT NOT NULL,
    original_name TEXT NOT NULL DEFAULT '',
    mime          TEXT NOT NULL DEFAULT '',
    sha256        TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    uri           TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL CHECK (status IN ('queued','processing','ready','failed')),
    error         TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    guide_json    TEXT NOT NULL DEFAULT '{}',
    char_count    INTEGER NOT NULL DEFAULT 0,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (notebook_id, sha256)
);
CREATE INDEX idx_sources_notebook ON sources(notebook_id, status);

CREATE TABLE source_files (
    source_id  TEXT PRIMARY KEY REFERENCES sources(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE chunks (
    id           TEXT PRIMARY KEY,
    source_id    TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    notebook_id  TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    position     INTEGER NOT NULL,
    text         TEXT NOT NULL,
    start_char   INTEGER,
    end_char     INTEGER,
    page         INTEGER,
    slide        INTEGER,
    t_start_ms   INTEGER,
    t_end_ms     INTEGER,
    section_path TEXT,
    token_count  INTEGER,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX idx_chunks_notebook ON chunks(notebook_id);
CREATE INDEX idx_chunks_source ON chunks(source_id, position);

CREATE TABLE chunk_vectors (
    chunk_id           TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    embedder_signature TEXT NOT NULL,
    dim                INTEGER NOT NULL,
    vector             BLOB NOT NULL,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (chunk_id, embedder_signature)
);

CREATE TABLE jobs (
    id           TEXT PRIMARY KEY,
    notebook_id  TEXT REFERENCES notebooks(id) ON DELETE CASCADE,
    type         TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed','cancelled')),
    progress     REAL NOT NULL DEFAULT 0,
    result_json  TEXT NOT NULL DEFAULT '{}',
    error        TEXT NOT NULL DEFAULT '',
    started_at   TEXT,
    finished_at  TEXT,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX idx_jobs_status ON jobs(status);
