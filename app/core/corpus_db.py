"""Transactional corpus persistence, replacing the whole-file JSON rewrite.

**The defect this exists to fix.** ``DocumentStore._save`` serialised the
entire in-memory corpus and renamed it over ``corpus.json``. Atomic per write,
and still lossy: two workers sharing a ``DATA_DIR`` each hold their own copy of
the corpus, so the second save overwrites the first with a snapshot that never
contained the first worker's document. Reproduced before this module existed —
two processes, two uploads, both returning 200, one document permanently gone:

    uploaded: ['a.md', 'b.md']  (both returned success)
    on disk : ['a.md']
    LOST    : ['b.md']

No lock could fix that. The in-process ``RLock`` does not span processes, and a
file lock would serialise whole-corpus rewrites into a bottleneck while still
losing data on any crash between read and write. The actual bug is the
*read-modify-write of the entire dataset*; the fix is to make each mutation
touch only its own rows, inside a transaction the database arbitrates.

**Why SQLite rather than a server database.** It is in the standard library, it
needs no daemon, and a reviewer can open the file with any SQLite client. The
properties that matter here — atomic multi-statement transactions, and
cross-process locking the kernel enforces rather than the application — are
exactly what the JSON file lacked. Scaling past one machine is a different
problem and would want Postgres; this project does not have that problem.

**WAL mode** is set because the default rollback journal makes writers block
readers. Under WAL a worker answering a question keeps reading while another
worker ingests a document, which is the common case in a multi-worker
deployment and the one the old design handled worst.

What this module deliberately does **not** do: keep each worker's in-memory
search index up to date. Durability and index freshness are separate problems.
A document written by worker 1 is safely on disk and will be loaded by every
worker at its next start, but worker 2's live index does not know about it
until then. See ``docs/multi-tenancy.md`` for why that limit is stated rather
than papered over.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Busy timeout for cross-process contention. SQLite returns SQLITE_BUSY
# immediately when another connection holds the write lock; this makes it wait
# and retry instead. Five seconds is far longer than any write here takes, so
# hitting it means something is genuinely wrong rather than merely contended.
BUSY_TIMEOUT_MS = 5000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    chunk_count  INTEGER NOT NULL,
    char_count   INTEGER NOT NULL,
    sha256       TEXT NOT NULL,
    uploaded_at  TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);

-- Enforces per-tenant deduplication in the database rather than in a Python
-- loop over the in-memory dict. Two workers racing to ingest the same bytes
-- now collide on this constraint instead of both succeeding and one winning.
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    position   INTEGER NOT NULL,
    text       TEXT NOT NULL,
    embedding  TEXT,
    -- ON DELETE CASCADE so removing a document cannot leave orphaned chunks
    -- that would still be indexed at the next boot.
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
"""


def _enable_wal(conn: sqlite3.Connection, attempts: int = 10) -> bool:
    """Switch the database to WAL, retrying briefly on contention.

    ``PRAGMA busy_timeout`` does **not** cover the lock that changing journal
    mode requires while another connection has the database open, so a plain
    ``PRAGMA journal_mode=WAL`` raises ``database is locked`` when several
    processes start at the same moment — precisely the ``--workers N`` case.
    Found by a stress test, not by reasoning: 8 workers starting together
    killed two of them outright.

    Failure is survivable and therefore not fatal. WAL is a concurrency
    optimisation; the rollback journal is still correct, just with writers
    blocking readers. Crashing a worker over a performance setting would be a
    worse outcome than running it in the slower mode, so this logs and
    continues.

    Note the mode is a property of the database file, not the connection: once
    any connection sets it, every later one inherits it, so in practice only
    the first connection after creation does real work here.
    """
    for attempt in range(attempts):
        try:
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
            if mode and str(mode[0]).lower() == "wal":
                return True
        except sqlite3.OperationalError:
            pass
        # Jittered backoff so retrying processes do not re-collide in lockstep.
        time.sleep(0.02 * (attempt + 1) * (0.5 + random.random()))

    logger.warning(
        "Could not switch %s to WAL mode; continuing with the default journal. "
        "Writers will block readers, which is slower but still correct.",
        getattr(conn, "_corpus_path", "the corpus database"),
    )
    return False


class CorpusDB:
    """SQLite-backed storage for one tenant's documents and chunks.

    One instance per tenant, over that tenant's own database file, so tenant
    isolation is unchanged: a query cannot reach another tenant's rows because
    it is not connected to that tenant's database.

    Connections are per-thread. ``sqlite3`` objects are not safe to share
    across threads, and these endpoints run in a threadpool, so a single shared
    connection would be a correctness bug rather than an optimisation.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        with self._init_lock:
            self._ensure_schema()

    # --- connection management ------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path),
            timeout=BUSY_TIMEOUT_MS / 1000.0,
            isolation_level=None,  # explicit transactions; see transaction()
        )
        conn.row_factory = sqlite3.Row
        # busy_timeout FIRST. Switching to WAL briefly needs an exclusive lock,
        # so if several processes start at once — which is exactly what
        # `--workers N` does — the losers get SQLITE_BUSY immediately unless a
        # timeout is already in effect. Setting it afterwards meant the one
        # statement most likely to contend was the one statement unprotected,
        # and workers crashed at startup instead of waiting their turn.
        conn.execute("PRAGMA busy_timeout={}".format(BUSY_TIMEOUT_MS))
        _enable_wal(conn)
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside one immediate transaction.

        ``BEGIN IMMEDIATE`` rather than the default deferred begin. A deferred
        transaction takes the write lock only at the first write, so two
        processes can both read, both decide to act on what they read, and then
        one fails at commit — the read-modify-write race this module exists to
        remove, reintroduced one level down. Taking the write lock up front
        makes the whole read-decide-write sequence serialised across processes.
        """
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # --- schema ----------------------------------------------------------

    def _ensure_schema(self) -> None:
        conn = self._conn
        # executescript() commits any open transaction and runs its own, so the
        # CREATE statements are deliberately outside the transaction below.
        # Every one is IF NOT EXISTS, which SQLite applies atomically, so
        # concurrent creators are safe here.
        conn.executescript(_SCHEMA)

        # The version row needs an actual transaction. Read-then-insert is a
        # check-then-act race: several workers starting together all saw no
        # row and all tried to insert, and the losers died on the primary key.
        # The upsert makes a concurrent insert a no-op rather than an error,
        # and BEGIN IMMEDIATE serialises the read that follows it.
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('version', ?) "
                "ON CONFLICT(key) DO NOTHING",
                (str(SCHEMA_VERSION),),
            )

        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()
        if row is not None and int(row["value"]) > SCHEMA_VERSION:
            # Refuse rather than guess. A newer schema may store columns this
            # code would silently drop on its next write.
            raise RuntimeError(
                "Corpus at {} was written by a newer version (schema {} > {}). "
                "Upgrade the application or point DATA_DIR elsewhere.".format(
                    self.path, row["value"], SCHEMA_VERSION
                )
            )

    # --- reads -----------------------------------------------------------

    def embedding_signature(self) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_signature'"
        ).fetchone()
        return row["value"] if row else None

    def set_embedding_signature(self, signature: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('embedding_signature', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (signature,),
            )

    def all_documents(self) -> List[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM documents ORDER BY uploaded_at, doc_id"
            ).fetchall()
        )

    def chunks_for(self, doc_id: str) -> List[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY position",
                (doc_id,),
            ).fetchall()
        )

    def find_by_sha256(self, digest: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM documents WHERE sha256 = ?", (digest,)
        ).fetchone()

    def document_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]

    # --- writes ----------------------------------------------------------

    def insert_document(
        self,
        meta: Dict[str, object],
        chunks: Sequence[Dict[str, object]],
    ) -> None:
        """Insert a document and its chunks as one atomic unit.

        Either both land or neither does. The old code could persist a document
        row and then fail while writing chunks, leaving a document that
        retrieves nothing.
        """
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO documents "
                "(doc_id, name, size_bytes, chunk_count, char_count, sha256, "
                " uploaded_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    meta["doc_id"],
                    meta["name"],
                    meta["size_bytes"],
                    meta["chunk_count"],
                    meta["char_count"],
                    meta["sha256"],
                    meta["uploaded_at"],
                    json.dumps(meta.get("metadata") or {}, ensure_ascii=False),
                ),
            )
            conn.executemany(
                "INSERT INTO chunks (chunk_id, doc_id, position, text, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        c["chunk_id"],
                        meta["doc_id"],
                        c["position"],
                        c["text"],
                        json.dumps(c["embedding"]) if c.get("embedding") else None,
                    )
                    for c in chunks
                ],
            )

    def update_metadata(self, doc_id: str, metadata: Dict[str, str]) -> bool:
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE documents SET metadata = ? WHERE doc_id = ?",
                (json.dumps(metadata, ensure_ascii=False), doc_id),
            )
            return cur.rowcount > 0

    def update_embeddings(self, vectors: Dict[str, Optional[List[float]]]) -> None:
        """Persist freshly computed vectors for existing chunks.

        Used after a backend change re-embeds the corpus, so the next boot is
        fast. Touches only the embedding column: a concurrent metadata update
        or a delete of another document is unaffected.
        """
        if not vectors:
            return
        with self.transaction() as conn:
            conn.executemany(
                "UPDATE chunks SET embedding = ? WHERE chunk_id = ?",
                [
                    (json.dumps(vec) if vec else None, chunk_id)
                    for chunk_id, vec in vectors.items()
                ],
            )

    def delete_document(self, doc_id: str) -> bool:
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            # Chunks go via ON DELETE CASCADE.
            return cur.rowcount > 0

    def clear(self) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM chunks")


def migrate_json_corpus(json_path: Path, db: CorpusDB) -> int:
    """Import a legacy ``corpus.json`` into ``db``. Returns documents imported.

    The JSON file is left in place. An upgrade that deletes the only copy of
    the data before anyone has confirmed the new path works is not a migration,
    it is a gamble — and the file is small enough that keeping it costs little.

    Idempotent: documents already present are skipped, so a half-finished
    migration can simply be run again.
    """
    if not json_path.exists():
        return 0

    try:
        raw = json.loads(json_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        logger.warning("Could not parse %s; nothing to migrate.", json_path)
        return 0

    if not isinstance(raw, dict) or not isinstance(raw.get("documents"), list):
        return 0

    existing = {row["doc_id"] for row in db.all_documents()}
    imported = 0

    for entry in raw["documents"]:
        try:
            if not isinstance(entry, dict):
                continue
            meta = dict(entry["meta"])
            doc_id = str(meta["doc_id"])
            if doc_id in existing:
                continue

            metadata = meta.pop("metadata", None) or {}
            chunks = [
                {
                    "chunk_id": str(c["chunk_id"]),
                    "position": int(c["position"]),
                    "text": str(c["text"]),
                    "embedding": c.get("embedding"),
                }
                for c in entry.get("chunks", [])
                if isinstance(c, dict)
            ]
            meta["metadata"] = metadata
            db.insert_document(meta, chunks)
            imported += 1
        except (KeyError, ValueError, TypeError, sqlite3.DatabaseError):
            # One unreadable entry must not abort the rest, matching the
            # salvage behaviour the JSON loader already had.
            logger.warning("Skipping an unreadable document during migration.", exc_info=True)

    if imported:
        signature = raw.get("embedding_signature")
        if isinstance(signature, str):
            db.set_embedding_signature(signature)
        logger.info(
            "Migrated %d document(s) from %s into %s. The JSON file is kept as "
            "a backup and is no longer read.",
            imported,
            json_path.name,
            db.path.name,
        )

    return imported
