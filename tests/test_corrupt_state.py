"""Malformed on-disk state must never prevent the server from starting.

`DocumentStore` and `PersonaStore` are both constructed at import time in
`app/main.py`. An uncaught exception in either loader is therefore not a
request-time 500 — it is a server that will not boot at all. These tests pin
that behaviour for every shape of corruption.
"""

from __future__ import annotations

import json

import pytest

from app.core.persona import Persona, PersonaStore
from app.core.store import DocumentStore


# --------------------------------------------------------------- persona

@pytest.mark.parametrize(
    "content",
    [
        "[1, 2, 3]",                      # a JSON array, not an object
        '"just a string"',                # a bare JSON string
        "null",                           # JSON null
        "42",                             # a number
        "{ truncated",                    # invalid JSON
        "",                               # empty file
        '{"greeting": null}',             # right shape, null field
        '{"name": null, "topic": null}',  # null where strings are expected
        '{"suggested_questions": "not a list"}',
        '{"suggested_questions": [1, 2, {"x": 1}]}',
        '{"unknown_field": "ignored"}',
    ],
)
def test_malformed_persona_file_falls_back_to_defaults(tmp_path, content):
    (tmp_path / "persona.json").write_text(content, "utf-8")

    store = PersonaStore(tmp_path)  # must not raise

    persona = store.current
    assert isinstance(persona.name, str)
    assert isinstance(persona.topic, str)
    assert isinstance(persona.style, str)
    assert isinstance(persona.suggested_questions, list)
    # The greeting is what actually crashed before: it calls .strip().
    assert isinstance(persona.effective_greeting(), str)


def test_persona_from_dict_rejects_non_dict():
    assert Persona.from_dict([1, 2, 3]).name == Persona().name
    assert Persona.from_dict("string").name == Persona().name
    assert Persona.from_dict(None).name == Persona().name


def test_persona_from_dict_keeps_valid_fields():
    persona = Persona.from_dict({"name": "Bot", "topic": "widgets", "bogus": 1})
    assert persona.name == "Bot"
    assert persona.topic == "widgets"
    assert persona.style == Persona().style  # untouched default


def test_valid_persona_still_round_trips(tmp_path):
    store = PersonaStore(tmp_path)
    store.set(Persona(name="Helper", topic="returns", greeting="Hi!"))

    reloaded = PersonaStore(tmp_path)
    assert reloaded.current.name == "Helper"
    assert reloaded.current.effective_greeting() == "Hi!"


# ---------------------------------------------------------------- corpus

@pytest.mark.parametrize(
    "content",
    [
        "{ not valid json",
        "[1, 2, 3]",                              # array instead of object
        "null",
        '"a string"',
        "",
        '{"documents": "not a list"}',
        '{"documents": [null, 42, "text"]}',      # junk entries
        '{"documents": [{}]}',                    # entry with no meta
        '{"documents": [{"meta": {}}]}',          # meta missing every field
        '{"documents": [{"meta": {"doc_id": "x"}}]}',  # meta partially filled
        '{"no_documents_key": true}',
    ],
)
def test_malformed_corpus_file_does_not_prevent_startup(tmp_path, content):
    (tmp_path / "corpus.json").write_text(content, "utf-8")

    store = DocumentStore(data_dir=tmp_path)  # must not raise

    assert store.is_empty
    assert store.list_documents() == []
    # The store must still be usable afterwards.
    store.add("recovery.md", b"The final exam is cumulative.")
    assert not store.is_empty


def test_document_with_malformed_chunks_is_skipped(tmp_path):
    """A bad chunk must not take down the whole corpus."""
    good = {
        "meta": {
            "doc_id": "good1",
            "name": "good.md",
            "size_bytes": 10,
            "chunk_count": 1,
            "char_count": 10,
            "sha256": "a" * 64,
            "uploaded_at": "2026-01-01T00:00:00+00:00",
        },
        "chunks": [{"chunk_id": "good1-0", "position": 0, "text": "exam policy"}],
    }
    bad = {
        "meta": {
            "doc_id": "bad1",
            "name": "bad.md",
            "size_bytes": 10,
            "chunk_count": 1,
            "char_count": 10,
            "sha256": "b" * 64,
            "uploaded_at": "2026-01-01T00:00:00+00:00",
        },
        "chunks": [{"position": 0}],  # missing chunk_id and text
    }
    (tmp_path / "corpus.json").write_text(
        json.dumps({"version": 1, "documents": [good, bad]}), "utf-8"
    )

    store = DocumentStore(data_dir=tmp_path)

    names = [d.name for d in store.list_documents()]
    assert "good.md" in names, "A salvageable document must survive"
    assert "bad.md" not in names
    assert store.index.search("exam policy")


def test_partially_indexed_document_is_not_left_behind(tmp_path):
    """A document whose chunks fail validation must not be half-registered."""
    entry = {
        "meta": {
            "doc_id": "d1",
            "name": "doc.md",
            "size_bytes": 10,
            "chunk_count": 2,
            "char_count": 10,
            "sha256": "c" * 64,
            "uploaded_at": "2026-01-01T00:00:00+00:00",
        },
        "chunks": [
            {"chunk_id": "d1-0", "position": 0, "text": "first chunk"},
            {"position": 1},  # invalid: raises partway through
        ],
    }
    (tmp_path / "corpus.json").write_text(
        json.dumps({"version": 1, "documents": [entry]}), "utf-8"
    )

    store = DocumentStore(data_dir=tmp_path)

    # Either fully loaded or not at all — never a document with orphaned chunks.
    assert store.is_empty
    assert len(store.index) == 0


# ----------------------------------------------------------- atomic write

def test_temp_file_name_is_unique_per_write(tmp_path):
    """A fixed temp name lets two processes clobber each other's corpus."""
    store = DocumentStore(data_dir=tmp_path)
    store.add("a.md", b"first document content")
    store.add("b.md", b"second document content")

    # No temp files should survive a successful write.
    assert list(tmp_path.glob("*.tmp")) == []
    assert json.loads((tmp_path / "corpus.json").read_text("utf-8"))["documents"]


def test_corpus_is_readable_after_many_writes(tmp_path):
    store = DocumentStore(data_dir=tmp_path)
    for i in range(10):
        store.add("doc{}.md".format(i), "content number {}".format(i).encode())

    reloaded = DocumentStore(data_dir=tmp_path)
    assert len(reloaded.list_documents()) == 10
