"""API surface added by this project: tagging, facets, scoped chat, /api/search.

Every test here is offline — ``/api/search`` never calls the LLM, and the chat
tests exercise paths that refuse before generation.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A fresh app instance per test, with an isolated data directory."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    import app.config as config
    import app.core.embedding as embedding

    config.reset_settings()
    embedding.reset_embedder()

    import app.main
    import importlib

    importlib.reload(app.main)
    with TestClient(app.main.app) as c:
        yield c

    config.reset_settings()
    embedding.reset_embedder()


def upload(client, name, text, metadata=None):
    files = {"files": (name, text.encode("utf-8"), "text/markdown")}
    data = {"metadata": json.dumps(metadata)} if metadata else {}
    return client.post("/api/documents", files=files, data=data)


SYLLABUS = (
    "# CS 1428 Syllabus\n\n"
    "The final exam is held in week sixteen. Late work loses ten percent "
    "per day. Office hours are Tuesdays at two o'clock.\n"
)
RETURNS = (
    "# Returns Policy\n\n"
    "Footwear may be returned within thirty days of delivery for a full "
    "refund, provided the boots are unworn.\n"
)


# ------------------------------------------------------------------ tagging


def test_upload_accepts_metadata(client):
    response = upload(client, "syllabus.md", SYLLABUS, {"course": "CS1428"})
    assert response.status_code == 200
    assert response.json()["uploaded"][0]["metadata"] == {"course": "cs1428"}


def test_upload_without_metadata_has_empty_tags(client):
    response = upload(client, "syllabus.md", SYLLABUS)
    assert response.json()["uploaded"][0]["metadata"] == {}


def test_malformed_metadata_json_is_rejected(client):
    files = {"files": ("a.md", b"content here", "text/markdown")}
    response = client.post(
        "/api/documents", files=files, data={"metadata": "{not json"}
    )
    assert response.status_code == 400


def test_non_object_metadata_is_rejected(client):
    files = {"files": ("a.md", b"content here", "text/markdown")}
    response = client.post(
        "/api/documents", files=files, data={"metadata": '["a", "b"]'}
    )
    assert response.status_code == 400


def test_metadata_can_be_replaced_after_upload(client):
    doc_id = upload(client, "syllabus.md", SYLLABUS).json()["uploaded"][0]["doc_id"]
    response = client.put(
        "/api/documents/{}/metadata".format(doc_id),
        json={"metadata": {"course": "CS2308"}},
    )
    assert response.status_code == 200
    assert response.json()["metadata"] == {"course": "cs2308"}


def test_metadata_update_on_unknown_document_is_404(client):
    response = client.put(
        "/api/documents/nope/metadata", json={"metadata": {"a": "b"}}
    )
    assert response.status_code == 404


def test_invalid_metadata_on_update_is_400(client):
    doc_id = upload(client, "syllabus.md", SYLLABUS).json()["uploaded"][0]["doc_id"]
    response = client.put(
        "/api/documents/{}/metadata".format(doc_id),
        json={"metadata": {"course": ["a", "b"]}},
    )
    assert response.status_code == 400


# ------------------------------------------------------------------- facets


def test_facets_are_empty_without_tags(client):
    upload(client, "syllabus.md", SYLLABUS)
    assert client.get("/api/facets").json()["facets"] == {}


def test_facets_count_documents_per_value(client):
    upload(client, "a.md", SYLLABUS, {"course": "CS1428"})
    upload(client, "b.md", RETURNS, {"course": "CS1428"})
    assert client.get("/api/facets").json()["facets"]["course"] == {"cs1428": 2}


def test_status_includes_facets_and_retrieval_config(client):
    upload(client, "a.md", SYLLABUS, {"course": "CS1428"})
    body = client.get("/api/status").json()
    assert body["facets"]["course"] == {"cs1428": 1}
    assert body["retrieval"]["hybrid_enabled"] is True
    assert body["retrieval"]["embedding_backend"] == "hashing"


# ------------------------------------------------------------------- search


def test_search_runs_without_an_api_key(client):
    """The whole retrieval stack must be gradeable with no key and no cost."""
    upload(client, "syllabus.md", SYLLABUS)
    response = client.post("/api/search", json={"query": "when is the final exam"})
    assert response.status_code == 200
    assert response.json()["hits"]


def test_search_exposes_per_retriever_ranks(client):
    upload(client, "syllabus.md", SYLLABUS)
    hit = client.post(
        "/api/search", json={"query": "when is the final exam"}
    ).json()["hits"][0]
    assert hit["retrievers"]
    assert hit["lexical_rank"] is not None or hit["dense_rank"] is not None


def test_search_respects_a_metadata_scope(client):
    upload(client, "syllabus.md", SYLLABUS, {"course": "CS1428"})
    upload(client, "returns.md", RETURNS, {"dept": "support"})

    body = client.post(
        "/api/search",
        json={"query": "refund for boots", "scope": {"course": "CS1428"}},
    ).json()

    assert body["documents_in_scope"] == 1
    assert body["scope_description"] == "course is cs1428"
    assert all(h["doc_name"] == "syllabus.md" for h in body["hits"])


def test_search_reports_empty_scope(client):
    upload(client, "syllabus.md", SYLLABUS, {"course": "CS1428"})
    body = client.post(
        "/api/search",
        json={"query": "final exam", "scope": {"course": "NOPE"}},
    ).json()
    assert body["decision"] == "empty_scope"
    assert body["hits"] == []


def test_search_rejects_a_malformed_filter(client):
    upload(client, "syllabus.md", SYLLABUS)
    response = client.post(
        "/api/search",
        json={"query": "final exam", "scope": {"course": {"startswith": "CS"}}},
    )
    assert response.status_code == 400


def test_search_top_k_is_bounded(client):
    upload(client, "syllabus.md", SYLLABUS)
    assert client.post(
        "/api/search", json={"query": "exam", "top_k": 0}
    ).status_code == 422


def test_search_requires_a_query(client):
    assert client.post("/api/search", json={"query": ""}).status_code == 422


# --------------------------------------------------------------------- chat


def test_chat_refuses_out_of_scope_without_a_key(client):
    """Layer 1 is deterministic: no key needed to refuse."""
    upload(client, "syllabus.md", SYLLABUS)
    body = client.post(
        "/api/chat", json={"message": "Who won the Super Bowl in 1998?"}
    ).json()
    assert body["in_scope"] is False
    assert body["decision"] == "out_of_scope"


def test_chat_reports_an_empty_scope_distinctly(client):
    """An unanswerable filter is a different problem from an off-topic question."""
    upload(client, "syllabus.md", SYLLABUS, {"course": "CS1428"})
    body = client.post(
        "/api/chat",
        json={"message": "when is the final exam", "scope": {"course": "NOPE"}},
    ).json()

    assert body["decision"] == "empty_scope"
    assert body["documents_in_scope"] == 0
    assert "no documents match" in body["answer"].lower()


def test_chat_rejects_a_malformed_filter(client):
    upload(client, "syllabus.md", SYLLABUS)
    response = client.post(
        "/api/chat",
        json={"message": "hello", "scope": {"course": {"bogus": "x"}}},
    )
    assert response.status_code == 400


def test_chat_without_a_key_returns_503_when_generation_is_needed(client):
    """In-scope questions need the LLM; the failure must be explicable."""
    upload(client, "syllabus.md", SYLLABUS)
    response = client.post(
        "/api/chat", json={"message": "when is the final exam"}
    )
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_refusal_mentions_the_active_scope(client):
    upload(client, "syllabus.md", SYLLABUS, {"course": "CS1428"})
    upload(client, "returns.md", RETURNS, {"dept": "support"})

    body = client.post(
        "/api/chat",
        json={
            "message": "Who won the Super Bowl in 1998?",
            "scope": {"course": "CS1428"},
        },
    ).json()

    assert body["in_scope"] is False
    assert "course is cs1428" in body["answer"].lower()
