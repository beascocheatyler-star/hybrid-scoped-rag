-- M6: Studio artifacts and per-user study progress.
CREATE TABLE artifacts (
    id              TEXT PRIMARY KEY,
    notebook_id     TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    created_by      TEXT NOT NULL REFERENCES users(id),
    type            TEXT NOT NULL CHECK (type IN ('report','data_table','mind_map','flashcards','quiz','audio_overview','slide_deck','infographic')),
    title           TEXT NOT NULL DEFAULT '',
    params_json     TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL CHECK (status IN ('queued','running','ready','failed','cancelled')),
    prompt_used     TEXT NOT NULL DEFAULT '',
    content_json    TEXT NOT NULL DEFAULT '{}',
    file_path       TEXT NOT NULL DEFAULT '',
    source_ids_json TEXT NOT NULL DEFAULT '[]',
    job_id          TEXT,
    generated_by    TEXT NOT NULL DEFAULT '',
    error           TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX idx_artifacts_notebook ON artifacts(notebook_id, created_at);

CREATE TABLE study_progress (
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    item_id     TEXT NOT NULL,
    state       TEXT NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    answer      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (user_id, artifact_id, item_id)
);
