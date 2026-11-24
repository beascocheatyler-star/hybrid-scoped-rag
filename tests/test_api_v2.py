"""HTTP tests for /api/v2 (notebooks, sources, search, jobs) and v1 mapping."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

EX = Path(__file__).resolve().parent.parent / "examples"
SYL = (EX / "course_ta" / "cs1428_syllabus.md").read_bytes()
SUP = (EX / "customer_care" / "returns_and_shipping.md").read_bytes()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import app.config as config
    import app.core.embedding as embedding

    config.reset_settings()
    embedding.reset_embedder()
    import app.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        c.main = main  # type: ignore[attr-defined]
        yield c
    main.services.close()
    config.reset_settings()
    embedding.reset_embedder()


def new_nb(c, title="Notebook"):
    r = c.post("/api/v2/notebooks", json={"title": title})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def upload(c, nb, name, data, metadata=None, wait=True):
    files = {"files": (name, data, "text/markdown")}
    form = {"metadata": json.dumps(metadata)} if metadata else {}
    return c.post("/api/v2/notebooks/{}/sources?wait={}".format(nb, str(wait).lower()),
                  files=files, data=form)


def test_me_and_default_notebook(client):
    assert client.get("/api/v2/me").json() == {
        "id": "local", "display_name": "Local user", "single_user_mode": True}
    nbs = client.get("/api/v2/notebooks").json()
    assert [n["id"] for n in nbs] == ["default"] and nbs[0]["is_default"]


def test_notebook_crud(client):
    nb = new_nb(client, "Biology")
    assert client.get("/api/v2/notebooks/" + nb).json()["title"] == "Biology"
    r = client.patch("/api/v2/notebooks/" + nb, json={"title": "Bio 101", "emoji": "🧬"})
    assert (r.json()["title"], r.json()["emoji"]) == ("Bio 101", "🧬")
    assert client.patch("/api/v2/notebooks/" + nb, json={"title": "  "}).status_code == 400
    assert client.delete("/api/v2/notebooks/" + nb).status_code == 204
    assert client.get("/api/v2/notebooks/" + nb).status_code == 404
    assert client.delete("/api/v2/notebooks/default").status_code == 409


def test_unknown_notebook_404_everywhere(client):
    for method, path in [("get", "/api/v2/notebooks/zzz"), ("get", "/api/v2/notebooks/zzz/sources"),
                         ("get", "/api/v2/notebooks/zzz/facets"), ("delete", "/api/v2/notebooks/zzz")]:
        assert getattr(client, method)(path).status_code == 404
    assert client.post("/api/v2/notebooks/zzz/search", json={"query": "x"}).status_code == 404


def test_upload_returns_jobs_and_ingests(client):
    nb = new_nb(client)
    r = upload(client, nb, "s.md", SYL, {"course": "CS1428"})
    assert r.status_code == 202, r.text
    q = r.json()["queued"][0]
    job = client.get("/api/v2/jobs/" + q["job_id"]).json()
    assert job["status"] == "succeeded" and job["type"] == "ingest"
    src = client.get("/api/v2/notebooks/{}/sources/{}".format(nb, q["source_id"])).json()
    assert src["status"] == "ready" and src["metadata"] == {"course": "cs1428"}
    assert src["has_original"] and src["chunk_count"] > 0
    assert client.get("/api/v2/notebooks/{}/facets".format(nb)).json() == {"course": {"cs1428": 1}}


def test_original_file_and_content(client):
    nb = new_nb(client)
    sid = upload(client, nb, "s.md", SYL).json()["queued"][0]["source_id"]
    f = client.get("/api/v2/notebooks/{}/sources/{}/file".format(nb, sid))
    assert f.status_code == 200 and f.content == SYL
    content = client.get("/api/v2/notebooks/{}/sources/{}/content".format(nb, sid)).json()
    assert "Foundations of Computer Science" in content["text"]


def test_text_source(client):
    nb = new_nb(client)
    r = client.post("/api/v2/notebooks/{}/sources?wait=true".format(nb),
                    json={"kind": "text", "title": "Lab notes",
                          "text": "Lab three covers pointers and arrays.", "metadata": {"week": "3"}})
    assert r.status_code == 202, r.text
    sid = r.json()["queued"][0]["source_id"]
    src = client.get("/api/v2/notebooks/{}/sources/{}".format(nb, sid)).json()
    assert (src["kind"], src["title"], src["status"]) == ("text", "Lab notes", "ready")
    assert client.post("/api/v2/notebooks/{}/sources".format(nb),
                       json={"kind": "url", "text": "x"}).status_code == 400
    assert client.post("/api/v2/notebooks/{}/sources".format(nb),
                       json={"kind": "text", "text": "  "}).status_code == 400


def test_duplicate_and_unsupported_are_skipped(client):
    nb = new_nb(client)
    upload(client, nb, "s.md", SYL)
    r = client.post("/api/v2/notebooks/{}/sources".format(nb), files=[
        ("files", ("again.md", SYL, "text/markdown")),
        ("files", ("r.md", SUP, "text/markdown")),
        ("files", ("x.exe", b"MZ", "application/octet-stream")),
    ])
    body = r.json()
    assert [q["name"] for q in body["queued"]] == ["r.md"]
    reasons = {s["name"]: s["reason"] for s in body["skipped"]}
    assert "already" in reasons["again.md"] and "Unsupported" in reasons["x.exe"]
    bad = client.post("/api/v2/notebooks/{}/sources".format(nb),
                      files={"files": ("y.exe", b"MZ", "application/octet-stream")})
    assert bad.status_code == 400


def test_failed_ingest_visible_on_source(client):
    nb = new_nb(client)
    r = upload(client, nb, "broken.pdf", b"not a pdf")
    q = r.json()["queued"][0]
    assert client.get("/api/v2/jobs/" + q["job_id"]).json()["status"] == "failed"
    src = client.get("/api/v2/notebooks/{}/sources/{}".format(nb, q["source_id"])).json()
    assert src["status"] == "failed" and src["error"]


def test_patch_and_delete_source(client):
    nb = new_nb(client)
    sid = upload(client, nb, "s.md", SYL).json()["queued"][0]["source_id"]
    base = "/api/v2/notebooks/{}/sources/{}".format(nb, sid)
    r = client.patch(base, json={"title": "Syllabus", "metadata": {"term": "Spring2026"}})
    assert (r.json()["title"], r.json()["metadata"]) == ("Syllabus", {"term": "spring2026"})
    hits = client.post("/api/v2/notebooks/{}/search".format(nb), json={"query": "final exam"}).json()["hits"]
    assert hits[0]["doc_name"] == "Syllabus"
    assert client.delete(base).status_code == 204
    assert client.get(base).status_code == 404
    assert client.delete(base).status_code == 404


def test_search_is_notebook_scoped_and_keyless(client):
    a, b = new_nb(client, "A"), new_nb(client, "B")
    upload(client, a, "r.md", SUP)
    upload(client, b, "s.md", SYL)
    rb = client.post("/api/v2/notebooks/{}/search".format(b),
                     json={"query": "How long do I have to return boots for a refund?"}).json()
    assert all(h["doc_name"] == "s.md" for h in rb["hits"])
    assert rb["decision"] == "out_of_scope"
    ra = client.post("/api/v2/notebooks/{}/search".format(a),
                     json={"query": "How long do I have to return an item?"}).json()
    assert ra["decision"] in {"in_scope", "borderline"}


def test_search_source_ids_and_scope(client):
    nb = new_nb(client)
    s1 = upload(client, nb, "s.md", SYL, {"course": "cs1428"}).json()["queued"][0]["source_id"]
    upload(client, nb, "r.md", SUP, {"dept": "support"})
    url = "/api/v2/notebooks/{}/search".format(nb)
    r = client.post(url, json={"query": "refund", "source_ids": [s1]}).json()
    assert all(h["doc_id"] == s1 for h in r["hits"])
    empty = client.post(url, json={"query": "final exam", "scope": {"course": "math"}}).json()
    assert empty["decision"] == "empty_scope"
    assert client.post(url, json={"query": "x", "scope": {"a": {"bad": 1}}}).status_code == 400


def test_copy_notebook_api(client):
    nb = new_nb(client, "Orig")
    upload(client, nb, "s.md", SYL)
    r = client.post("/api/v2/notebooks/{}/copy".format(nb), json={"title": "Clone"})
    assert r.status_code == 201 and r.json()["source_count"] == 1
    clone = r.json()["id"]
    hits = client.post("/api/v2/notebooks/{}/search".format(clone), json={"query": "final exam"}).json()["hits"]
    assert hits


def test_v1_routes_operate_on_default_notebook(client):
    r = client.post("/api/documents", files={"files": ("s.md", SYL, "text/markdown")})
    assert r.status_code == 200
    srcs = client.get("/api/v2/notebooks/default/sources").json()
    assert [s["title"] for s in srcs] == ["s.md"]
    other = new_nb(client)
    assert client.get("/api/v2/notebooks/{}/sources".format(other)).json() == []
    # v1 search/chat only see the Default notebook.
    upload(client, other, "r.md", SUP)
    v1 = client.post("/api/search", json={"query": "refund boots warranty"}).json()
    assert all(h["doc_name"] == "s.md" for h in v1["hits"])


def test_cancel_job_api(client):
    nb = new_nb(client)
    st = client.main.registry.store(nb)
    src = st.create_source("s.md", SYL)
    job = client.main.jobs.enqueue("ingest", {"source_id": src.id}, notebook_id=nb, start=False)
    r = client.post("/api/v2/jobs/{}/cancel".format(job.id))
    assert r.json()["status"] == "cancelled"
    assert client.get("/api/v2/jobs/nope").status_code == 404
