"""M2: anchored chunking, new source kinds, SSRF guard, Source Guides, labels."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from hypothesis import given, settings as hsettings, strategies as st

import app.core.extract as extract_mod
import app.core.fetch as fetch_mod
from app.config import Settings
from app.core.chunking import chunk_spans, chunk_text
from app.core.embedding import HashingEmbedder
from app.core.extract import ExtractionError, extract_doc
from app.core.fetch import FetchError, Fetched, check_url
from app.core.guide import AUTO_LABEL_MIN, build_guide
from app.core.llm import AnthropicLLM, set_llm_override
from app.core.services import build_services
from tests.fakes import FakeLLM
from tests.fixtures_gen import make_docx, make_epub, make_pdf, make_pptx

EX = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def svc(tmp_path):
    s = Settings()
    s.data_dir = tmp_path
    s.source_guides = False
    services = build_services(s, HashingEmbedder(dimensions=128))
    yield services
    services.close()


@pytest.fixture(autouse=True)
def _reset_hooks():
    yield
    extract_mod.set_transcriber(None)
    fetch_mod.set_transcript_fetcher(None)
    set_llm_override(None)


def assert_round_trip(store, source_id):
    text = store.extracted_text(source_id)
    chunks = store.source_chunks(source_id)
    assert chunks
    for c in chunks:
        assert text[c["start_char"]:c["end_char"]] == c["text"]
    return text, chunks


# ------------------------------------------------------------- chunk spans


@pytest.mark.parametrize("path", sorted(EX.rglob("*.md")), ids=lambda p: p.name)
def test_chunk_spans_match_legacy_chunking(path):
    t = path.read_text("utf-8")
    spans, legacy = chunk_spans(t), chunk_text(t)
    assert len(spans) == len(legacy)
    for sp, lc in zip(spans, legacy):
        assert t[sp.start:sp.end] == sp.text
        assert sp.text.split() == lc.split()  # same words => same BM25 tokens


_words = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=0, max_size=12)


@hsettings(max_examples=150, deadline=None)
@given(st.lists(st.tuples(_words, st.sampled_from([" ", "\n", "\n\n", ". ", "  \n \n"])),
                max_size=200),
       st.integers(min_value=20, max_value=400), st.integers(min_value=0, max_value=300))
def test_chunk_spans_round_trip_property(parts, size, overlap):
    text = "".join(w + sep for w, sep in parts)
    for sp in chunk_spans(text, size, overlap):
        assert text[sp.start:sp.end] == sp.text
        assert sp.text.strip()


# ------------------------------------------------------------- source kinds


def test_markdown_sections(svc):
    st_ = svc.registry.default_store()
    doc = st_.add("cs1428_syllabus.md", (EX / "course_ta" / "cs1428_syllabus.md").read_bytes())
    _, chunks = assert_round_trip(st_, doc.doc_id)
    assert all(c["section_path"] for c in chunks)
    d = extract_doc("s.md", (EX / "course_ta" / "cs1428_syllabus.md").read_bytes())
    pos = d.text.index("10 percentage points per calendar day")
    assert d.section_at(pos).endswith("Programming assignments > Late work policy")
    hit = st_.index.search("late work penalty")[0]
    assert hit.chunk.start_char is not None and hit.chunk.section_path


def test_pdf_pages_and_citation_page(svc):
    pdf = make_pdf(["Introduction to sorting algorithms.",
                    "Quicksort partitions the array around a pivot element.",
                    "Mergesort divides and merges sorted halves."])
    st_ = svc.registry.default_store()
    doc = st_.add("algos.pdf", pdf)
    text, chunks = assert_round_trip(st_, doc.doc_id)
    assert "[page" not in text
    # Small chunk size so each page lands in its own chunk.
    st2 = svc.registry.store(svc.registry.create("small").id)
    st2.chunk_size, st2.overlap = 60, 0
    d2 = st2.add("algos.pdf", pdf)
    hit = st2.index.search("quicksort pivot partition")[0]
    assert hit.chunk.page == 2
    assert {c["page"] for c in st2.source_chunks(d2.doc_id)} == {1, 2, 3}


def test_docx_headings(svc):
    data = make_docx([("h1", "Course Policies"), ("p", "Attendance is required in every lab."),
                      ("h2", "Grading"), ("p", "Homework counts for forty percent of the grade.")])
    st_ = svc.registry.default_store()
    doc = st_.add("policy.docx", data)
    _, chunks = assert_round_trip(st_, doc.doc_id)
    d = extract_doc("policy.docx", data)
    assert d.section_at(d.text.index("Homework")) == "Course Policies > Grading"


def test_pptx_slides_and_notes(svc):
    data = make_pptx([(["Photosynthesis", "Light reactions make ATP"], ["Mention chlorophyll"]),
                      (["Calvin cycle", "Carbon fixation uses RuBisCO"], [])])
    d = extract_doc("bio.pptx", data)
    assert "Speaker notes: Mention chlorophyll" in d.text
    assert d.slide_at(d.text.index("RuBisCO")) == 2
    st_ = svc.registry.store(svc.registry.create("p").id)
    st_.chunk_size, st_.overlap = 60, 0
    doc = st_.add("bio.pptx", data)
    _, chunks = assert_round_trip(st_, doc.doc_id)
    assert st_.index.search("RuBisCO carbon fixation")[0].chunk.slide == 2


def test_epub(svc):
    data = make_epub("Tiny Book", [("Chapter One", "The river flooded the valley in spring."),
                                   ("Chapter Two", "Farmers rebuilt the levees by autumn.")])
    d = extract_doc("book.epub", data)
    assert d.title == "Tiny Book"
    assert d.section_at(d.text.index("levees")) == "Chapter Two"
    st_ = svc.registry.default_store()
    assert_round_trip(st_, st_.add("book.epub", data).doc_id)


def test_html_file(svc):
    html = (b"<html><head><title>T</title><script>evil()</script></head><body><nav>menu</nav>"
            b"<h1>Heading</h1><p>Body text about volcanoes and magma.</p></body></html>")
    d = extract_doc("page.html", html)
    assert "evil" not in d.text and "menu" not in d.text and "volcanoes" in d.text
    st_ = svc.registry.default_store()
    assert_round_trip(st_, st_.add("page.html", html).doc_id)


def test_image_ocr_with_fake(svc, monkeypatch):
    monkeypatch.setattr(extract_mod, "_ocr_image_bytes", lambda data: "Whiteboard: mitosis has four phases.")
    st_ = svc.registry.default_store()
    doc = st_.add("board.png", b"\x89PNG fake")
    assert st_.get_source(doc.doc_id).kind == "image"
    assert_round_trip(st_, doc.doc_id)


def test_missing_ocr_dependency_is_a_clear_source_error(svc, monkeypatch):
    import builtins

    real = builtins.__import__

    def no_tesseract(name, *a, **k):
        if name in ("pytesseract", "PIL"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_tesseract)
    nb = svc.registry.create("x")
    st_ = svc.registry.store(nb.id)
    src = st_.create_source("scan.png", b"\x89PNG fake")
    job = svc.jobs.wait(svc.jobs.enqueue("ingest", {"source_id": src.id}, notebook_id=nb.id).id)
    assert job.status == "failed"
    s = st_.get_source(src.id)
    assert s.status == "failed" and "OCR extras" in s.error


def test_audio_with_fake_transcriber(svc):
    extract_mod.set_transcriber(lambda data, name: [
        (0, 4000, "Welcome to the lecture on thermodynamics."),
        (4000, 9000, "Entropy always increases in an isolated system."),
    ])
    st_ = svc.registry.store(svc.registry.create("a").id)
    st_.chunk_size, st_.overlap = 50, 0
    doc = st_.add("lecture.mp3", b"ID3fakeaudio")
    assert st_.get_source(doc.doc_id).kind == "audio"
    _, chunks = assert_round_trip(st_, doc.doc_id)
    hit = st_.index.search("entropy isolated system")[0]
    assert (hit.chunk.t_start_ms, hit.chunk.t_end_ms) == (4000, 9000)


def test_youtube_with_fake_transcript(svc):
    fetch_mod.set_transcript_fetcher(lambda vid: [
        (0, 2000, "Today we talk about volcanoes."),
        (2000, 5000, "Magma rises through the crust."),
        (65000, 70000, "Later, we cover earthquakes and fault lines."),
    ])
    st_ = svc.registry.store(svc.registry.create("y").id)
    src = st_.create_remote_source("youtube", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    doc = st_.process_source(src.id)
    s = st_.get_source(doc.doc_id)
    assert s.kind == "youtube" and s.title == "YouTube video dQw4w9WgXcQ"
    _, chunks = assert_round_trip(st_, doc.doc_id)
    assert chunks[0]["t_start_ms"] == 0
    assert st_.original_path(doc.doc_id) is not None


def test_url_source_with_fake_fetch(svc, monkeypatch):
    html = (b'<html><head><title>Volcano facts</title><link rel="canonical" href="https://ex.org/v"></head>'
            b"<body><article><h2>Magma</h2><p>Basaltic magma has low viscosity.</p></article></body></html>")
    monkeypatch.setattr(fetch_mod, "_resolver", lambda host: ["93.184.216.34"])
    import app.core.notebooks as nbmod

    monkeypatch.setattr(nbmod, "fetch_bytes", lambda url: Fetched(url, "text/html", html))
    st_ = svc.registry.store(svc.registry.create("u").id)
    src = st_.create_remote_source("url", "https://ex.org/volcano?ref=1")
    doc = st_.process_source(src.id)
    s = st_.get_source(doc.doc_id)
    assert (s.kind, s.title, s.uri) == ("url", "Volcano facts", "https://ex.org/v")
    assert_round_trip(st_, doc.doc_id)
    assert st_.original_path(doc.doc_id).read_bytes() == html


def test_max_source_words(svc):
    st_ = svc.registry.default_store()
    st_.max_source_words = 10
    with pytest.raises(ExtractionError, match="MAX_SOURCE_WORDS"):
        st_.add("long.txt", b"word " * 50)


# ------------------------------------------------------------- SSRF guard


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/admin", "http://[::1]/", "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/", "http://192.168.1.1/", "http://0.0.0.0/", "file:///etc/passwd",
    "ftp://example.com/x", "http://example.com:22/", "http:///nohost",
])
def test_ssrf_guard_rejects(url, monkeypatch):
    monkeypatch.setattr(fetch_mod, "_resolver", lambda host: ["93.184.216.34"])
    with pytest.raises(FetchError):
        check_url(url)


def test_ssrf_guard_resolves_names(monkeypatch):
    monkeypatch.setattr(fetch_mod, "_resolver", lambda host: ["127.0.0.1"])
    with pytest.raises(FetchError, match="non-public"):
        check_url("http://localhost.example/")
    monkeypatch.setattr(fetch_mod, "_resolver", lambda host: ["93.184.216.34", "10.1.1.1"])
    with pytest.raises(FetchError):  # any private address taints the host
        check_url("http://mixed.example/")
    monkeypatch.setattr(fetch_mod, "_resolver", lambda host: ["93.184.216.34"])
    assert check_url("https://example.com/page").hostname == "example.com"


def test_redirect_to_private_is_blocked(monkeypatch):
    import urllib.error

    monkeypatch.setattr(fetch_mod, "_resolver",
                        lambda host: ["10.0.0.9"] if host == "internal.example" else ["93.184.216.34"])

    class Opener:
        def open(self, req, timeout=0):
            from email.message import Message
            h = Message()
            h["Location"] = "http://internal.example/secret"
            raise urllib.error.HTTPError(req.full_url, 302, "Found", h, None)

    monkeypatch.setattr(fetch_mod, "_opener", Opener())
    with pytest.raises(FetchError, match="non-public"):
        fetch_mod.fetch_bytes("http://public.example/")


def test_url_to_localhost_rejected_by_api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app.config as config
    import app.core.embedding as embedding

    config.reset_settings()
    embedding.reset_embedder()
    import app.main as main

    importlib.reload(main)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        r = c.post("/api/v2/notebooks/default/sources",
                   json={"kind": "url", "url": "http://127.0.0.1:8000/api/status"})
        assert r.status_code == 400 and "non-public" in r.json()["detail"]
    main.services.close()
    config.reset_settings()


# ------------------------------------------------------------- guides & labels


def test_extractive_guide_has_valid_citations():
    t = (EX / "public_domain" / "darwin_origin_ch3.md").read_text("utf-8")
    chunks = [{"id": "c{}".format(i), "text": s.text} for i, s in enumerate(chunk_spans(t))]
    g = build_guide(chunks, "Darwin")
    assert 120 <= len(g["summary"].split()) <= 260
    assert 5 <= len(g["topics"]) <= 10 and 3 <= len(g["questions"]) <= 5
    assert g["citations"] and all(c["chunk_id"].startswith("c") for c in g["citations"])
    assert g["generated_by"] == "extractive"


def test_llm_guide_drops_hallucinated_markers():
    reply = json.dumps({"summary": "Basalt is runny [1]. Granite is stiff [9].",
                        "topics": ["magma", "basalt", "granite", "viscosity", "eruptions"],
                        "questions": ["Why is basalt runny?", "What is viscosity?", "Which magma erupts?"]})
    fake = FakeLLM(reply=reply)
    llm = AnthropicLLM("claude-opus-5-5", client=fake)
    g = build_guide([{"id": "a", "text": "Basaltic magma has low viscosity."}], "Magma", llm)
    assert g["generated_by"] == "llm"
    assert g["citations"] == [{"marker": 1, "chunk_id": "a"}]
    assert "[9]" not in g["summary"]
    call = fake.last
    assert "untrusted" in call["system"].lower()
    assert call["output_config"]["format"]["type"] == "json_schema"


def test_guide_job_and_auto_labels(tmp_path):
    s = Settings()
    s.data_dir = tmp_path
    svc = build_services(s, HashingEmbedder(dimensions=128))
    try:
        nb = svc.registry.create("Labels")
        st_ = svc.registry.store(nb.id)
        texts = {
            "v1.md": "Volcanoes erupt magma. Volcanoes form at plate boundaries. Volcano magma cools.",
            "v2.md": "Magma chambers feed volcanoes. Volcanoes and magma shape islands over time.",
            "v3.md": "Volcano monitoring tracks magma movement and gas emissions near volcanoes.",
            "b1.md": "Bread baking needs flour, water and yeast. Bread dough rises with yeast.",
            "b2.md": "Sourdough bread uses wild yeast. Bread crust forms in hot ovens.",
        }
        jobs = []
        for name, body in texts.items():
            src = st_.create_source(name, (body * 3).encode())
            jobs.append(svc.jobs.enqueue("ingest", {"source_id": src.id}, notebook_id=nb.id))
        for j in jobs:
            svc.jobs.wait(j.id)
        svc.jobs.drain()
        svc.jobs.drain()
        srcs = {x.title: x for x in st_.list_sources()}
        assert all(x.guide.get("summary") for x in srcs.values())
        labels = {t: x.metadata.get("label") for t, x in srcs.items()}
        assert all(labels.values()), labels
        assert labels["v1.md"] == labels["v2.md"] == labels["v3.md"]
        assert labels["b1.md"] == labels["b2.md"] != labels["v1.md"]
        assert "label" in st_.facets()
        # A user edit survives re-labelling.
        st_.set_metadata(srcs["v1.md"].id, {"label": "my volcano notes"})
        svc.jobs.wait(svc.jobs.enqueue("autolabel", {}, notebook_id=nb.id).id)
        assert st_.get_source(srcs["v1.md"].id).metadata["label"] == "my volcano notes"
        assert st_.get_source(srcs["v2.md"].id).metadata["label"] == labels["v2.md"]
    finally:
        svc.close()


def test_content_highlight_api(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SOURCE_GUIDES", "false")
    import app.config as config
    import app.core.embedding as embedding

    config.reset_settings()
    embedding.reset_embedder()
    import app.main as main

    importlib.reload(main)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        r = c.post("/api/v2/notebooks/default/sources?wait=true",
                   files={"files": ("s.md", (EX / "course_ta" / "cs1428_syllabus.md").read_bytes(),
                                    "text/markdown")})
        sid = r.json()["queued"][0]["source_id"]
        hit = c.post("/api/v2/notebooks/default/search", json={"query": "late work penalty"}).json()["hits"][0]
        body = c.get("/api/v2/notebooks/default/sources/{}/content".format(sid),
                     params={"highlight": hit["chunk_id"]}).json()
        h = body["highlight"]
        assert body["text"][h["start_char"]:h["end_char"]] == hit["text"]
        assert h["section_path"]
        assert c.get("/api/v2/notebooks/default/sources/{}/content".format(sid),
                     params={"highlight": "nope"}).status_code == 404
    main.services.close()
    config.reset_settings()
