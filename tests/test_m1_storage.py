"""M1: SQLite storage, migrations, notebooks, originals, jobs, legacy import."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from app.config import Settings
from app.core.db import Database, discover_migrations, pack_vector, unpack_vector
from app.core.embedding import HashingEmbedder
from app.core.metadata import ScopeFilter
from app.core.notebooks import (
    DEFAULT_NOTEBOOK_ID,
    NotebookNotFound,
    NotebookRegistry,
    SourceLimitError,
)
from app.core.services import build_services
from app.core.store import DocumentStore, DuplicateDocumentError

EX = Path(__file__).resolve().parent.parent / "examples"
SYL = (EX / "course_ta" / "cs1428_syllabus.md").read_bytes()
SUP = (EX / "customer_care" / "returns_and_shipping.md").read_bytes()


def make_settings(tmp_path, **kw) -> Settings:
    s = Settings()
    s.data_dir = tmp_path
    s.job_workers = 2
    for k, v in kw.items():
        setattr(s, k, v)
    return s


@pytest.fixture
def svc(tmp_path):
    s = build_services(make_settings(tmp_path), HashingEmbedder(dimensions=256))
    yield s
    s.close()


# ------------------------------------------------------------- migrations


def test_fresh_database_is_fully_migrated(tmp_path):
    db = Database(tmp_path / "x.sqlite3")
    assert db.schema_version == max(n for n, _ in discover_migrations())
    assert db.one("PRAGMA journal_mode")[0] == "wal"
    assert db.one("PRAGMA foreign_keys")[0] == 1


def test_migrations_are_idempotent(tmp_path):
    db = Database(tmp_path / "x.sqlite3")
    assert db.migrate() == []
    db.close()
    assert Database(tmp_path / "x.sqlite3").migrate() == []


def test_failed_migration_rolls_back(tmp_path):
    mig = tmp_path / "migs"
    mig.mkdir()
    (mig / "001_ok.sql").write_text("CREATE TABLE a (x INTEGER);")
    (mig / "002_bad.sql").write_text("CREATE TABLE b (x INTEGER); INSERT INTO nope VALUES (1);")
    with pytest.raises(sqlite3.OperationalError):
        Database(tmp_path / "y.sqlite3", migrations_dir=mig)
    conn = sqlite3.connect(str(tmp_path / "y.sqlite3"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "a" in tables and "b" not in tables
    assert [r[0] for r in conn.execute("SELECT version FROM schema_migrations")] == [1]


def test_bad_migration_name_is_rejected(tmp_path):
    (tmp_path / "1_x.sql").write_text("")
    with pytest.raises(ValueError):
        discover_migrations(tmp_path)


def test_vector_blob_round_trip():
    v = [0.25, -1.5, 3.0, 0.0]
    assert unpack_vector(pack_vector(v)) == v


# ------------------------------------------------------------- isolation


def test_notebook_isolation_acceptance(svc):
    """Roadmap M1 acceptance: A has both examples, B only the syllabus.
    Search in B never returns A's chunks."""
    reg = svc.registry
    a, b = reg.create("A"), reg.create("B")
    sa, sb = reg.store(a.id), reg.store(b.id)
    sa.add("cs1428_syllabus.md", SYL)
    sa.add("returns_and_shipping.md", SUP)
    sb.add("cs1428_syllabus.md", SYL)
    a_chunks = {c.chunk_id for c in sa.index._chunks}
    for q in ["refund footwear return shipping", "warranty claim", "final exam", "boots"]:
        for h in sb.index.search(q, top_k=50):
            assert h.chunk.chunk_id not in a_chunks
            assert h.chunk.doc_id in {d.doc_id for d in sb.list_documents()}


def test_idf_is_per_notebook(svc):
    reg = svc.registry
    a, b = reg.create("A"), reg.create("B")
    sb = reg.store(b.id)
    sb.add("cs1428_syllabus.md", SYL)
    before = [(h.chunk.chunk_id, h.normalized, h.score) for h in sb.index.search("final exam", top_k=6)]
    sa = reg.store(a.id)
    sa.add("returns_and_shipping.md", SUP)
    sa.add("x.md", b"final exam final exam final exam week sixteen")
    after = [(h.chunk.chunk_id, h.normalized, h.score) for h in sb.index.search("final exam", top_k=6)]
    assert before == after
    assert len(sb.index) == sb.total_chunks == sum(d.chunk_count for d in sb.list_documents())


def test_facets_are_per_notebook(svc):
    reg = svc.registry
    a, b = reg.create("A"), reg.create("B")
    reg.store(a.id).add("s.md", SYL, {"course": "cs1428"})
    reg.store(b.id).add("r.md", SUP, {"dept": "support"})
    assert reg.store(a.id).facets() == {"course": {"cs1428": 1}}
    assert reg.store(b.id).facets() == {"dept": {"support": 1}}


def test_duplicates_are_per_notebook(svc):
    reg = svc.registry
    a, b = reg.create("A"), reg.create("B")
    reg.store(a.id).add("s.md", SYL)
    reg.store(b.id).add("s.md", SYL)  # same bytes, other notebook: allowed
    with pytest.raises(DuplicateDocumentError):
        reg.store(a.id).add("again.md", SYL)


def test_unknown_notebook(svc):
    with pytest.raises(NotebookNotFound):
        svc.registry.store("nope")


def test_default_notebook_exists_and_cannot_be_deleted(svc):
    assert svc.registry.get(DEFAULT_NOTEBOOK_ID).title == "Default notebook"
    with pytest.raises(ValueError):
        svc.registry.delete(DEFAULT_NOTEBOOK_ID)


def test_source_limit(tmp_path):
    s = build_services(make_settings(tmp_path, max_sources_per_notebook=2),
                       HashingEmbedder(dimensions=64))
    nb = s.registry.create("Small")
    st = s.registry.store(nb.id)
    st.add("a.md", b"alpha beta gamma")
    st.add("b.md", b"delta epsilon zeta")
    with pytest.raises(SourceLimitError):
        st.add("c.md", b"eta theta iota")
    # The Default notebook (v1 API) is never limited.
    d = s.registry.default_store()
    for i in range(3):
        d.add("{}.md".format(i), "doc number {}".format(i).encode())
    s.close()


# ------------------------------------------------------------- persistence


def test_reload_reproduces_index(tmp_path):
    s1 = build_services(make_settings(tmp_path), HashingEmbedder(dimensions=256))
    nb = s1.registry.create("A")
    s1.registry.store(nb.id).add("s.md", SYL, {"course": "cs1428"})
    q1 = [(h.chunk.chunk_id, round(h.score, 9)) for h in s1.registry.store(nb.id).index.search("late work penalty")]
    s1.close()
    s2 = build_services(make_settings(tmp_path), HashingEmbedder(dimensions=256))
    st = s2.registry.store(nb.id)
    assert [d.name for d in st.list_documents()] == ["s.md"]
    assert st.facets() == {"course": {"cs1428": 1}}
    q2 = [(h.chunk.chunk_id, round(h.score, 9)) for h in st.index.search("late work penalty")]
    assert q1 == q2
    s2.close()


def test_embedder_change_reembeds_and_persists(tmp_path):
    s1 = build_services(make_settings(tmp_path), HashingEmbedder(dimensions=128))
    s1.registry.default_store().add("s.md", SYL)
    s1.close()
    s2 = build_services(make_settings(tmp_path), HashingEmbedder(dimensions=64))
    st = s2.registry.default_store()
    assert all(len(c.embedding) == 64 for c in st.index._chunks)
    sigs = {r[0] for r in s2.db.all("SELECT embedder_signature FROM chunk_vectors")}
    assert sigs == {"hashing:128", "hashing:64"}
    s2.close()


def test_original_is_kept_and_extracted_text_stored(svc):
    st = svc.registry.default_store()
    doc = st.add("s.md", SYL)
    assert st.original_path(doc.doc_id).read_bytes() == SYL
    assert "Foundations of Computer Science" in st.extracted_text(doc.doc_id)


def test_delete_cascades(svc):
    st = svc.registry.default_store()
    doc = st.add("s.md", SYL)
    d = st.source_dir(doc.doc_id)
    assert d.exists()
    assert st.delete(doc.doc_id)
    assert not d.exists()
    for table in ("chunks", "chunk_vectors", "source_files"):
        assert svc.db.one("SELECT COUNT(*) AS n FROM {}".format(table))["n"] == 0
    assert st.index.search("final exam") == []


def test_failed_v1_add_leaves_nothing(svc):
    st = svc.registry.default_store()
    with pytest.raises(Exception):
        st.add("broken.pdf", b"not a pdf")
    assert st.list_sources() == []


def test_copy_notebook(svc):
    reg = svc.registry
    a = reg.create("A", "desc", "📚")
    sa = reg.store(a.id)
    sa.add("s.md", SYL, {"course": "cs1428"})
    c = reg.copy(a.id)
    assert c.title == "Copy of A" and c.emoji == "📚"
    sc = reg.store(c.id)
    assert [d.name for d in sc.list_documents()] == ["s.md"]
    assert sc.facets() == sa.facets()
    hit_ids = {h.chunk.chunk_id for h in sc.index.search("final exam")}
    assert hit_ids and not (hit_ids & {x.chunk_id for x in sa.index._chunks})
    src = sc.list_sources()[0]
    assert sc.original_path(src.id).read_bytes() == SYL
    reg.delete(a.id)  # the copy is independent
    assert reg.store(c.id).index.search("final exam")


def test_notebook_update_and_list(svc):
    reg = svc.registry
    nb = reg.create("  Spaced   title ")
    assert nb.title == "Spaced title"
    nb = reg.update(nb.id, title="New", emoji="🧪")
    assert (nb.title, nb.emoji) == ("New", "🧪")
    assert {n.id for n in reg.list()} >= {DEFAULT_NOTEBOOK_ID, nb.id}
    with pytest.raises(ValueError):
        reg.update(nb.id, title="   ")


def test_rename_updates_citation_name(svc):
    st = svc.registry.default_store()
    doc = st.add("s.md", SYL)
    st.rename(doc.doc_id, "Syllabus Spring 2026")
    assert st.get(doc.doc_id).name == "Syllabus Spring 2026"
    assert all(h.chunk.doc_name == "Syllabus Spring 2026" for h in st.index.search("final exam"))


# ------------------------------------------------------------- jobs


def test_ingest_job_succeeds(svc):
    nb = svc.registry.create("A")
    st = svc.registry.store(nb.id)
    src = st.create_source("s.md", SYL)
    assert st.is_empty  # queued sources are not searchable
    job = svc.jobs.enqueue("ingest", {"source_id": src.id}, notebook_id=nb.id)
    job = svc.jobs.wait(job.id)
    assert job.status == "succeeded" and job.progress == 1.0
    assert job.result["chunk_count"] > 0
    assert st.get_source(src.id).status == "ready"
    assert st.index.search("final exam")


def test_ingest_job_failure_is_recorded(svc):
    nb = svc.registry.create("A")
    st = svc.registry.store(nb.id)
    src = st.create_source("broken.pdf", b"not a pdf at all")
    job = svc.jobs.wait(svc.jobs.enqueue("ingest", {"source_id": src.id}, notebook_id=nb.id).id)
    assert job.status == "failed" and job.error
    s = st.get_source(src.id)
    assert s.status == "failed" and s.error
    assert st.is_empty


def test_cancel_queued_job(svc):
    nb = svc.registry.create("A")
    st = svc.registry.store(nb.id)
    src = st.create_source("s.md", SYL)
    job = svc.jobs.enqueue("ingest", {"source_id": src.id}, notebook_id=nb.id, start=False)
    assert svc.jobs.cancel(job.id).status == "cancelled"
    svc.jobs._submit(job.id)  # a cancelled job never starts
    svc.jobs.drain()
    assert svc.jobs.get(job.id).status == "cancelled"
    assert st.get_source(src.id).status == "queued"


def test_cancel_running_job_at_checkpoint(svc):
    nb = svc.registry.create("A")
    st = svc.registry.store(nb.id)
    src = st.create_source("s.md", SYL)
    calls = {"n": 0}

    def cancelled():
        calls["n"] += 1
        return calls["n"] >= 2  # cancel after the first stage

    from app.core.notebooks import JobCancelled

    with pytest.raises(JobCancelled):
        st.process_source(src.id, cancelled=cancelled)
    assert st.get_source(src.id).status == "failed"
    assert st.is_empty


def test_jobs_survive_restart(tmp_path):
    """A job left 'running' by a crash is re-queued and completed on boot."""
    s1 = build_services(make_settings(tmp_path), HashingEmbedder(dimensions=64), recover_jobs=False)
    nb = s1.registry.create("A")
    st = s1.registry.store(nb.id)
    src = st.create_source("s.md", SYL)
    job = s1.jobs.enqueue("ingest", {"source_id": src.id}, notebook_id=nb.id, start=False)
    s1.db.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (job.id,))
    s1.db.execute("UPDATE sources SET status = 'processing' WHERE id = ?", (src.id,))
    s1.close()  # simulated crash

    s2 = build_services(make_settings(tmp_path), HashingEmbedder(dimensions=64))
    s2.jobs.drain()
    assert s2.jobs.get(job.id).status == "succeeded"
    st2 = s2.registry.store(nb.id)
    assert st2.get_source(src.id).status == "ready"
    assert st2.index.search("final exam")
    s2.close()


def test_unknown_job_type_rejected(svc):
    with pytest.raises(ValueError):
        svc.jobs.enqueue("nope", {})


# ------------------------------------------------------------- legacy import


def _write_v2_corpus(tmp_path):
    legacy = DocumentStore(tmp_path, embedder=HashingEmbedder(dimensions=256))
    legacy.add("s.md", SYL, {"course": "cs1428"})
    legacy.add("r.md", SUP, {"dept": "support"})
    return legacy


def test_import_v2_corpus(tmp_path):
    legacy = _write_v2_corpus(tmp_path)
    (tmp_path / "persona.json").write_text(json.dumps({"name": "TA Bot", "topic": "CS 1428"}))
    want = {d.doc_id: d.name for d in legacy.list_documents()}
    q = [(h.chunk.chunk_id, round(h.score, 6)) for h in legacy.index.search("final exam")]

    s = build_services(make_settings(tmp_path), HashingEmbedder(dimensions=256))
    st = s.registry.default_store()
    assert {d.doc_id: d.name for d in st.list_documents()} == want
    assert st.facets() == {"course": {"cs1428": 1}, "dept": {"support": 1}}
    assert [(h.chunk.chunk_id, round(h.score, 6)) for h in st.index.search("final exam")] == q
    assert not (tmp_path / "corpus.json").exists()
    backups = list(tmp_path.glob("legacy-backup-*"))
    assert len(backups) == 1
    assert json.loads((backups[0] / "corpus.json").read_text())["version"] == 2
    assert (backups[0] / "persona.json").exists()
    assert (tmp_path / "persona.json").exists()  # v1 persona routes still use it
    assert s.registry.get(DEFAULT_NOTEBOOK_ID).chat_config["persona"]["name"] == "TA Bot"
    assert s.import_summary["imported"] == 2
    s.close()

    # Second boot: nothing re-imported, nothing duplicated.
    s2 = build_services(make_settings(tmp_path), HashingEmbedder(dimensions=256))
    assert s2.import_summary is None
    assert len(s2.registry.default_store().list_documents()) == 2
    s2.close()


def test_import_v1_corpus(tmp_path):
    legacy = {
        "version": 1,
        "documents": [{
            "meta": {"doc_id": "abc123", "name": "old.md", "size_bytes": 50,
                     "chunk_count": 1, "char_count": 50, "sha256": "f" * 64,
                     "uploaded_at": "2025-01-01T00:00:00+00:00"},
            "chunks": [{"chunk_id": "abc123-0", "position": 0,
                        "text": "The final exam is in week sixteen."}],
        }],
    }
    (tmp_path / "corpus.json").write_text(json.dumps(legacy))
    s = build_services(make_settings(tmp_path), HashingEmbedder(dimensions=256))
    st = s.registry.default_store()
    assert [d.name for d in st.list_documents()] == ["old.md"]
    assert st.index.search("final exam")[0].chunk.chunk_id == "abc123-0"
    s.close()


def test_import_corrupt_corpus_does_not_crash_and_keeps_backup(tmp_path):
    (tmp_path / "corpus.json").write_text("{not json")
    s = build_services(make_settings(tmp_path), HashingEmbedder(dimensions=64))
    assert s.registry.default_store().is_empty
    backup = next(tmp_path.glob("legacy-backup-*"))
    assert (backup / "corpus.json").read_text() == "{not json"
    s.close()


# ------------------------------------------------------------- performance


def test_upload_time_does_not_grow_with_corpus_size(tmp_path):
    """The JSON store rewrote the whole corpus per upload (~310 ms at 600
    docs). SQLite inserts only the new rows. Generous bound for CI noise."""
    s = build_services(make_settings(tmp_path), HashingEmbedder(dimensions=64))
    st = s.registry.default_store()
    times = []
    for i in range(200):
        body = "Document {} covers topic {} in depth. ".format(i, i * 7).encode() * 20
        t0 = time.perf_counter()
        st.add("d{}.md".format(i), body)
        times.append(time.perf_counter() - t0)
    first, last = sum(times[:30]) / 30, sum(times[-30:]) / 30
    assert last < first * 3 + 0.005, (first, last)
    s.close()
