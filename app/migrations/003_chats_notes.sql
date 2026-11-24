-- M4: persistent chats, messages and notes.
CREATE TABLE chats (
    id                       TEXT PRIMARY KEY,
    notebook_id              TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    user_id                  TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title                    TEXT NOT NULL DEFAULT 'New chat',
    selected_source_ids_json TEXT NOT NULL DEFAULT 'null',
    scope_json               TEXT NOT NULL DEFAULT 'null',
    created_at               TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at               TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX idx_chats_owner ON chats(notebook_id, user_id, updated_at);

CREATE TABLE messages (
    id             TEXT PRIMARY KEY,
    chat_id        TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role           TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content        TEXT NOT NULL,
    decision       TEXT NOT NULL DEFAULT '',
    reason         TEXT NOT NULL DEFAULT '',
    retrieval_json TEXT NOT NULL DEFAULT '{}',
    citations_json TEXT NOT NULL DEFAULT '[]',
    usage_json     TEXT NOT NULL DEFAULT '{}',
    model          TEXT NOT NULL DEFAULT '',
    feedback       TEXT NOT NULL DEFAULT '' CHECK (feedback IN ('','up','down')),
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX idx_messages_chat ON messages(chat_id, created_at);

CREATE TABLE notes (
    id              TEXT PRIMARY KEY,
    notebook_id     TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    author_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title           TEXT NOT NULL DEFAULT 'Untitled note',
    body_md         TEXT NOT NULL DEFAULT '',
    citations_json  TEXT NOT NULL DEFAULT '[]',
    from_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    is_source       INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX idx_notes_notebook ON notes(notebook_id, updated_at);
