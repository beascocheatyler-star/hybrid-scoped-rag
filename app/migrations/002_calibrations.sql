-- M3: measured gate floors per embedder signature (scripts/measure_gate.py).
CREATE TABLE calibrations (
    embedder_signature TEXT PRIMARY KEY,
    relevance_floor    REAL NOT NULL,
    dense_rescue_floor REAL NOT NULL,
    measured_at        TEXT NOT NULL,
    report_path        TEXT NOT NULL DEFAULT '',
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
