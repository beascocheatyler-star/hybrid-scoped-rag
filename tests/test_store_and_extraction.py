"""Document extraction, ingestion, persistence, and deletion."""

from __future__ import annotations

import json

import pytest

from app.core.extract import ExtractionError, extract
from app.core.store import DocumentStore, DuplicateDocumentError


# ------------------------------------------------------------ extraction

def test_extract_plain_text():
    assert "hello" in extract("a.txt", b"hello world")


def test_extract_markdown():
    assert "Heading" in extract("a.md", b"# Heading\n\nBody text.")


def test_extract_handles_utf8():
    text = extract("a.txt", "café résumé".encode("utf-8"))
    assert "café" in text


def test_extract_falls_back_to_latin1():
    # Invalid UTF-8 should still decode rather than crash.
    assert extract("a.txt", b"caf\xe9")


def test_extract_csv_pairs_headers_with_values():
    data = b"name,role\nAlice,instructor\nBob,TA\n"
    text = extract("roster.csv", data)
    assert "name: Alice" in text
    assert "role: instructor" in text
    assert "Columns: name, role" in text


def test_extract_json_is_pretty_printed():
    text = extract("a.json", b'{"policy":{"late_days":3}}')
    assert "late_days" in text
    assert "\n" in text  # indented, not a single line


def test_extract_rejects_invalid_json():
    with pytest.raises(ExtractionError, match="Invalid JSON"):
        extract("a.json", b"{not json")


def test_extract_rejects_empty_file():
    with pytest.raises(ExtractionError, match="empty"):
        extract("a.txt", b"")


def test_extract_rejects_unsupported_extension():
    with pytest.raises(ExtractionError, match="Unsupported file type"):
        extract("image.png", b"\x89PNG\r\n")


def test_extract_rejects_whitespace_only_file():
    with pytest.raises(ExtractionError):
        extract("a.txt", b"   \n\t  \n ")


def test_extract_pdf_roundtrip(tmp_path):
    """Build a real PDF with pypdf, then read it back."""
    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    target = tmp_path / "blank.pdf"
    with open(target, "wb") as fh:
        writer.write(fh)

    # A blank page has no extractable text — the error must be actionable.
    with pytest.raises(ExtractionError, match="No selectable text"):
        extract("blank.pdf", target.read_bytes())


def test_extract_docx_roundtrip(tmp_path):
    docx = pytest.importorskip("docx")

    document = docx.Document()
    document.add_paragraph("The late work policy allows three late days.")
    target = tmp_path / "policy.docx"
    document.save(str(target))

    text = extract("policy.docx", target.read_bytes())
    assert "late days" in text


# ----------------------------------------------------------------- store

def test_add_indexes_the_document(empty_store):
    doc = empty_store.add("notes.md", b"The final exam is cumulative.")
    assert doc.chunk_count >= 1
    assert empty_store.total_chunks >= 1
    assert not empty_store.is_empty
    assert empty_store.index.search("final exam")


def test_add_records_metadata(empty_store):
    data = b"Some policy text about returns and refunds."
    doc = empty_store.add("policy.md", data)
    assert doc.name == "policy.md"
    assert doc.size_bytes == len(data)
    assert doc.char_count > 0
    assert len(doc.sha256) == 64
    assert doc.uploaded_at


def test_duplicate_content_is_rejected(empty_store):
    empty_store.add("a.md", b"identical content here")
    with pytest.raises(DuplicateDocumentError):
        empty_store.add("renamed.md", b"identical content here")


def test_duplicate_error_names_the_existing_document(empty_store):
    empty_store.add("original.md", b"content")
    try:
        empty_store.add("copy.md", b"content")
    except DuplicateDocumentError as exc:
        assert exc.existing.name == "original.md"
        assert "original.md" in str(exc)


def test_delete_removes_document_and_chunks(empty_store):
    doc = empty_store.add("a.md", b"The final exam is cumulative.")
    assert empty_store.delete(doc.doc_id) is True
    assert empty_store.is_empty
    assert empty_store.total_chunks == 0
    assert empty_store.index.search("final exam") == []


def test_delete_unknown_document_returns_false(empty_store):
    assert empty_store.delete("does-not-exist") is False


def test_clear_removes_everything(empty_store):
    empty_store.add("a.md", b"first document text")
    empty_store.add("b.md", b"second document text")
    empty_store.clear()
    assert empty_store.is_empty
    assert empty_store.list_documents() == []


def test_list_documents_is_ordered_by_upload_time(empty_store):
    empty_store.add("a.md", b"first")
    empty_store.add("b.md", b"second")
    names = [d.name for d in empty_store.list_documents()]
    assert names == ["a.md", "b.md"]


# ----------------------------------------------------------- persistence

def test_corpus_survives_a_restart(tmp_path):
    first = DocumentStore(data_dir=tmp_path)
    first.add("syllabus.md", b"The final exam is cumulative and worth 20%.")

    # A fresh store over the same directory simulates a process restart.
    second = DocumentStore(data_dir=tmp_path)
    assert not second.is_empty
    assert [d.name for d in second.list_documents()] == ["syllabus.md"]
    assert second.index.search("final exam"), "Index should be rebuilt from disk"


def test_deletion_persists_across_restart(tmp_path):
    first = DocumentStore(data_dir=tmp_path)
    doc = first.add("a.md", b"some content to index")
    first.delete(doc.doc_id)

    second = DocumentStore(data_dir=tmp_path)
    assert second.is_empty


def test_a_corrupt_corpus_file_does_not_prevent_startup(tmp_path):
    (tmp_path / "corpus.json").write_text("{ this is not valid json", "utf-8")
    store = DocumentStore(data_dir=tmp_path)  # must not raise
    assert store.is_empty


def test_corpus_is_persisted_to_the_database(tmp_path):
    import sqlite3

    store = DocumentStore(data_dir=tmp_path)
    store.add("a.md", b"content for the corpus")

    conn = sqlite3.connect(str(tmp_path / "corpus.db"))
    try:
        docs = conn.execute("SELECT name FROM documents").fetchall()
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()

    assert [d[0] for d in docs] == ["a.md"]
    assert chunks >= 1
