"""Embedding persistence: the cache must never serve vectors across spaces.

Vectors from two different embedders are not comparable. A stale cache would
not crash — it would rank confidently and wrongly, which is the worst failure
mode available. These tests pin the invalidation behaviour.
"""

from __future__ import annotations

import json
import sqlite3

from app.core.embedding import HashingEmbedder
from app.core.store import DocumentStore

TEXT = b"The final exam is held in week sixteen of the term."


def stored_vectors(path):
    """Every persisted embedding, decoded, in chunk order.

    Reads the database directly rather than through DocumentStore, so these
    tests still verify what actually reached disk rather than trusting the
    in-memory state they are meant to be checking.
    """
    conn = sqlite3.connect(str(path / "corpus.db"))
    try:
        rows = conn.execute(
            "SELECT embedding FROM chunks ORDER BY doc_id, position"
        ).fetchall()
    finally:
        conn.close()
    return [json.loads(r[0]) if r[0] else None for r in rows]


def stored_signature(path):
    conn = sqlite3.connect(str(path / "corpus.db"))
    try:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_signature'"
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def corrupt_first_vector(path, value):
    """Write a bad embedding straight into the database."""
    conn = sqlite3.connect(str(path / "corpus.db"))
    try:
        chunk_id = conn.execute(
            "SELECT chunk_id FROM chunks ORDER BY doc_id, position LIMIT 1"
        ).fetchone()[0]
        conn.execute(
            "UPDATE chunks SET embedding = ? WHERE chunk_id = ?",
            (json.dumps(value), chunk_id),
        )
        conn.commit()
    finally:
        conn.close()


def test_embeddings_are_persisted(tmp_path):
    store = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    store.add("syllabus.md", TEXT)

    assert len(stored_vectors(tmp_path)[0]) == 256


def test_corpus_records_the_embedding_signature(tmp_path):
    store = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    store.add("syllabus.md", TEXT)
    assert stored_signature(tmp_path) == "hashing:256"


def test_cached_vectors_are_reused_on_reload(tmp_path):
    first = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    first.add("syllabus.md", TEXT)
    original = first.index._chunks[0].embedding

    second = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    assert second.index._chunks[0].embedding == original


def test_changing_dimensions_invalidates_the_cache(tmp_path):
    first = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    first.add("syllabus.md", TEXT)

    second = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=512))
    assert len(second.index._chunks[0].embedding) == 512
    assert stored_signature(tmp_path) == "hashing:512"


def test_reembedded_corpus_is_saved_once(tmp_path):
    """Re-embedding at boot should persist, so the next boot is fast again."""
    first = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    first.add("syllabus.md", TEXT)

    DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=512))

    assert len(stored_vectors(tmp_path)[0]) == 512


def test_a_corrupt_vector_is_recomputed_not_trusted(tmp_path):
    store = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    store.add("syllabus.md", TEXT)

    corrupt_first_vector(tmp_path, [0.1, 0.2])  # wrong width

    reloaded = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    assert len(reloaded.index._chunks[0].embedding) == 256


def test_a_non_list_vector_is_recomputed(tmp_path):
    store = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    store.add("syllabus.md", TEXT)

    corrupt_first_vector(tmp_path, "not a vector")

    reloaded = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    assert len(reloaded.index._chunks[0].embedding) == 256
    assert reloaded.index.search("final exam")


def test_a_version_1_corpus_still_loads(tmp_path):
    """Corpora written by the predecessor project must upgrade in place."""
    legacy = {
        "version": 1,
        "saved_at": "2026-01-01T00:00:00+00:00",
        "documents": [
            {
                "meta": {
                    "doc_id": "abc123",
                    "name": "syllabus.md",
                    "size_bytes": 51,
                    "chunk_count": 1,
                    "char_count": 51,
                    "sha256": "0" * 64,
                    "uploaded_at": "2026-01-01T00:00:00+00:00",
                },
                "chunks": [
                    {
                        "chunk_id": "abc123-0",
                        "position": 0,
                        "text": "The final exam is held in week sixteen.",
                    }
                ],
            }
        ],
    }
    (tmp_path / "corpus.json").write_text(json.dumps(legacy), "utf-8")

    store = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))

    assert len(store.list_documents()) == 1
    assert store.list_documents()[0].metadata == {}
    # It had no vectors; they must have been computed on load.
    assert store.index._chunks[0].embedding is not None
    assert store.index.search("final exam")


def test_metadata_survives_a_restart(tmp_path):
    first = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    first.add("syllabus.md", TEXT, metadata={"course": "CS1428"})

    second = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    assert second.list_documents()[0].metadata == {"course": "cs1428"}
    # And it must reach the chunks, or scoped search would silently fail.
    assert second.index._chunks[0].metadata == {"course": "cs1428"}


def test_metadata_update_survives_a_restart(tmp_path):
    first = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    doc = first.add("syllabus.md", TEXT)
    first.set_metadata(doc.doc_id, {"course": "CS2308"})

    second = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    assert second.index._chunks[0].metadata == {"course": "cs2308"}


def test_facets_survive_a_restart(tmp_path):
    first = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    first.add("syllabus.md", TEXT, metadata={"course": "CS1428"})

    second = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    assert second.facets() == {"course": {"cs1428": 1}}
