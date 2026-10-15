"""Transactional persistence: the multi-worker data loss and its fixes.

The headline test is ``test_concurrent_processes_do_not_lose_documents``. It
spawns real processes, because the bug it pins is invisible to threads: the old
design lost data precisely because each *process* held its own copy of the
corpus, and in-process locks cannot see across that boundary.

Three defects found by stress-testing rather than by reasoning are pinned here
too. All three only appear when several processes start at the same instant,
which no single-process test would ever have produced.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sqlite3
import time

import pytest

from app.core.corpus_db import CorpusDB, migrate_json_corpus
from app.core.store import DocumentStore, DuplicateDocumentError

ROOT = pathlib.Path(__file__).resolve().parent.parent

# "spawn" matches how uvicorn --workers starts workers, and unlike "fork" it
# cannot accidentally inherit an open database handle that would mask the race.
CTX = mp.get_context("spawn")


def _upload(root, data_dir, name, text, start_at):
    """Run in a child process: open the corpus, wait, then ingest.

    Must be module-level and importable, because "spawn" re-imports this
    module in the child rather than inheriting memory.
    """
    import sys
    import time as _time

    sys.path.insert(0, root)
    from app.core.store import DocumentStore

    store = DocumentStore(data_dir=data_dir, chunk_size=1200, overlap=200)
    # Every worker loads the same (empty) corpus *before* the barrier, which is
    # what made the old whole-file rewrite clobber the others.
    delay = start_at - _time.time()
    if delay > 0:
        _time.sleep(delay)
    store.add(name, text.encode("utf-8"))


# ------------------------------------------------- the lost-update defect

def test_concurrent_processes_do_not_lose_documents(tmp_path):
    """Several processes ingesting at once must all survive.

    Against the JSON corpus this failed every time: two processes, two
    successful uploads, one document permanently absent from disk. The client
    that lost its document still got a 200.
    """
    count = 6
    names = ["doc{}.md".format(i) for i in range(count)]
    start_at = time.time() + 1.5

    procs = [
        CTX.Process(
            target=_upload,
            args=(
                str(ROOT),
                str(tmp_path),
                name,
                "unique content {} ".format(i) * 40,
                start_at,
            ),
        )
        for i, name in enumerate(names)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=45)

    assert all(p.exitcode == 0 for p in procs), (
        "worker(s) crashed: {}".format([p.exitcode for p in procs])
    )

    conn = sqlite3.connect(str(tmp_path / "corpus.db"))
    try:
        stored = sorted(r[0] for r in conn.execute("SELECT name FROM documents"))
        orphans = conn.execute(
            "SELECT COUNT(*) FROM chunks "
            "WHERE doc_id NOT IN (SELECT doc_id FROM documents)"
        ).fetchone()[0]
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()

    assert stored == sorted(names), "lost: {}".format(sorted(set(names) - set(stored)))
    assert orphans == 0
    assert integrity == "ok"


def test_many_processes_can_open_the_corpus_simultaneously(tmp_path):
    """Opening the database must not crash on a cold start.

    Two defects lived here, both only visible when processes start together:
    ``PRAGMA busy_timeout`` was set *after* ``journal_mode=WAL`` so the one
    statement most likely to contend was unprotected, and the schema version
    row was inserted after a read, which is a check-then-act race that killed
    the losers on the primary key.
    """
    start_at = time.time() + 1.0
    procs = [
        CTX.Process(
            target=_upload,
            args=(
                str(ROOT),
                str(tmp_path),
                "p{}.md".format(i),
                "content {} ".format(i) * 30,
                start_at,
            ),
        )
        for i in range(8)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=45)

    assert [p.exitcode for p in procs] == [0] * 8


# --------------------------------------------------------- transactions

def test_a_failed_insert_leaves_no_partial_document(tmp_path):
    """Document and chunks land together or not at all."""
    db = CorpusDB(tmp_path / "corpus.db")
    meta = {
        "doc_id": "abc123",
        "name": "a.md",
        "size_bytes": 10,
        "chunk_count": 2,
        "char_count": 10,
        "sha256": "0" * 64,
        "uploaded_at": "2026-01-01T00:00:00+00:00",
        "metadata": {},
    }
    bad_chunks = [
        {"chunk_id": "abc123-0", "position": 0, "text": "fine", "embedding": None},
        # Duplicate primary key: the second insert fails mid-transaction.
        {"chunk_id": "abc123-0", "position": 1, "text": "boom", "embedding": None},
    ]

    with pytest.raises(sqlite3.IntegrityError):
        db.insert_document(meta, bad_chunks)

    assert db.all_documents() == []
    assert db.chunks_for("abc123") == []


def test_deleting_a_document_removes_its_chunks(tmp_path):
    store = DocumentStore(data_dir=tmp_path)
    doc = store.add("a.md", b"some content to chunk and index")

    conn = sqlite3.connect(str(tmp_path / "corpus.db"))
    try:
        before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert before > 0
        store.delete(doc.doc_id)
        after = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()

    assert after == 0, "ON DELETE CASCADE left orphaned chunks"


def test_duplicate_detection_survives_a_restart(tmp_path):
    """Deduplication reads the database, not just this process's memory."""
    first = DocumentStore(data_dir=tmp_path)
    first.add("a.md", b"identical bytes here")

    second = DocumentStore(data_dir=tmp_path)
    with pytest.raises(DuplicateDocumentError):
        second.add("different-name.md", b"identical bytes here")


def test_delete_reports_false_when_another_worker_already_removed_it(tmp_path):
    """Existence is decided by the database, not by local state."""
    a = DocumentStore(data_dir=tmp_path)
    doc = a.add("a.md", b"content for deletion")

    b = DocumentStore(data_dir=tmp_path)
    assert b.delete(doc.doc_id) is True
    # `a` still has it in memory, but it is gone from the database.
    assert a.delete(doc.doc_id) is False


# ----------------------------------------------------------- migration

def test_a_json_corpus_is_migrated_on_first_boot(tmp_path):
    legacy = {
        "version": 2,
        "embedding_signature": "hashing:1024",
        "documents": [
            {
                "meta": {
                    "doc_id": "abc123",
                    "name": "syllabus.md",
                    "size_bytes": 38,
                    "chunk_count": 1,
                    "char_count": 38,
                    "sha256": "0" * 64,
                    "uploaded_at": "2026-01-01T00:00:00+00:00",
                    "metadata": {"course": "cs1428"},
                },
                "chunks": [
                    {
                        "chunk_id": "abc123-0",
                        "position": 0,
                        "text": "The final exam is held in week sixteen.",
                        "embedding": None,
                    }
                ],
            }
        ],
    }
    (tmp_path / "corpus.json").write_text(json.dumps(legacy), "utf-8")

    store = DocumentStore(data_dir=tmp_path)

    assert [d.name for d in store.list_documents()] == ["syllabus.md"]
    assert store.list_documents()[0].metadata == {"course": "cs1428"}
    assert store.index.search("final exam")
    # The JSON file is kept as a backup rather than deleted.
    assert (tmp_path / "corpus.json").exists()


def test_migration_is_idempotent(tmp_path):
    db = CorpusDB(tmp_path / "corpus.db")
    legacy = {
        "documents": [
            {
                "meta": {
                    "doc_id": "abc123",
                    "name": "a.md",
                    "size_bytes": 5,
                    "chunk_count": 1,
                    "char_count": 5,
                    "sha256": "1" * 64,
                    "uploaded_at": "2026-01-01T00:00:00+00:00",
                },
                "chunks": [
                    {"chunk_id": "abc123-0", "position": 0, "text": "hello"}
                ],
            }
        ]
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(legacy), "utf-8")

    assert migrate_json_corpus(path, db) == 1
    assert migrate_json_corpus(path, db) == 0
    assert len(db.all_documents()) == 1


def test_a_corrupt_json_corpus_does_not_prevent_startup(tmp_path):
    (tmp_path / "corpus.json").write_text("{ not json at all", "utf-8")
    store = DocumentStore(data_dir=tmp_path)
    assert store.list_documents() == []
    # And the store is still usable.
    store.add("a.md", b"fresh content after a bad migration")
    assert len(store.list_documents()) == 1


def test_one_unreadable_entry_does_not_abort_the_migration(tmp_path):
    legacy = {
        "documents": [
            {"meta": {"doc_id": "bad"}},  # missing required fields
            {
                "meta": {
                    "doc_id": "good",
                    "name": "good.md",
                    "size_bytes": 5,
                    "chunk_count": 1,
                    "char_count": 5,
                    "sha256": "2" * 64,
                    "uploaded_at": "2026-01-01T00:00:00+00:00",
                },
                "chunks": [{"chunk_id": "good-0", "position": 0, "text": "hello"}],
            },
        ]
    }
    (tmp_path / "corpus.json").write_text(json.dumps(legacy), "utf-8")

    store = DocumentStore(data_dir=tmp_path)
    assert [d.name for d in store.list_documents()] == ["good.md"]


# ------------------------------------------------------------- schema

def test_a_newer_schema_is_refused(tmp_path):
    db = CorpusDB(tmp_path / "corpus.db")
    with db.transaction() as conn:
        conn.execute("UPDATE schema_meta SET value = '999' WHERE key = 'version'")

    with pytest.raises(RuntimeError, match="newer version"):
        CorpusDB(tmp_path / "corpus.db")


def test_tenant_databases_are_separate_files(tmp_path):
    a = DocumentStore(data_dir=tmp_path / "acme")
    b = DocumentStore(data_dir=tmp_path / "globex")

    a.add("acme.md", b"acme private content here")
    b.add("globex.md", b"globex private content here")

    assert [d.name for d in a.list_documents()] == ["acme.md"]
    assert [d.name for d in b.list_documents()] == ["globex.md"]
    assert (tmp_path / "acme" / "corpus.db").exists()
    assert (tmp_path / "globex" / "corpus.db").exists()
