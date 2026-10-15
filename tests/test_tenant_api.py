"""End-to-end tenant isolation over HTTP.

``test_tenancy.py`` proves the stores are separate objects. That is necessary
but not sufficient: isolation is only real if the *API* enforces it, since a
single endpoint that forgets to resolve the caller's tenant would breach it
regardless of how well the core is layered. These tests go through the HTTP
surface and try to cross the boundary at every endpoint that touches data.

Everything here is offline. No ANTHROPIC_API_KEY is set, so chat requests that
would reach the model are confined to paths that refuse before generation.
"""

from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient

ACME_KEY = "acme-secret-key-1234"
GLOBEX_KEY = "globex-secret-key-99"

ACME_AUTH = {"X-API-Key": ACME_KEY}
GLOBEX_AUTH = {"X-API-Key": GLOBEX_KEY}


def _build_client(tmp_path, monkeypatch, *, multi_tenant: bool):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MULTI_TENANT", "1" if multi_tenant else "0")
    monkeypatch.setenv(
        "TENANTS",
        "acme:Acme University:{},globex:Globex Retail:{}".format(ACME_KEY, GLOBEX_KEY),
    )

    import app.config as config
    import app.core.embedding as embedding

    config.reset_settings()
    embedding.reset_embedder()

    import app.main

    importlib.reload(app.main)
    return app.main, config, embedding


@pytest.fixture
def client(tmp_path, monkeypatch):
    main, config, embedding = _build_client(tmp_path, monkeypatch, multi_tenant=True)
    with TestClient(main.app) as c:
        yield c
    config.reset_settings()
    embedding.reset_embedder()


@pytest.fixture
def single_tenant_client(tmp_path, monkeypatch):
    main, config, embedding = _build_client(tmp_path, monkeypatch, multi_tenant=False)
    with TestClient(main.app) as c:
        yield c
    config.reset_settings()
    embedding.reset_embedder()


def upload(client, name, text, auth, metadata=None):
    files = {"files": (name, text.encode("utf-8"), "text/markdown")}
    data = {"metadata": json.dumps(metadata)} if metadata else {}
    return client.post("/api/documents", files=files, data=data, headers=auth)


@pytest.fixture
def loaded(client, syllabus_text, support_text):
    assert upload(client, "cs1428_syllabus.md", syllabus_text, ACME_AUTH).status_code == 200
    assert (
        upload(client, "returns_and_shipping.md", support_text, GLOBEX_AUTH).status_code
        == 200
    )
    return client


# ------------------------------------------------------- authentication

def test_requests_without_a_key_are_rejected(client):
    for method, path in [
        ("get", "/api/documents"),
        ("get", "/api/status"),
        ("get", "/api/facets"),
        ("get", "/api/persona"),
    ]:
        response = getattr(client, method)(path)
        assert response.status_code == 401, (
            "{} {} should require a key, got {}".format(
                method.upper(), path, response.status_code
            )
        )


def test_an_invalid_key_is_rejected(client):
    response = client.get("/api/documents", headers={"X-API-Key": "not-a-real-key"})
    assert response.status_code == 401


def test_chat_and_search_require_a_key(client):
    assert client.post("/api/chat", json={"message": "hello"}).status_code == 401
    assert client.post("/api/search", json={"query": "hello"}).status_code == 401


def test_a_valid_key_is_accepted(client):
    response = client.get("/api/documents", headers=ACME_AUTH)
    assert response.status_code == 200


def test_status_reports_the_authenticated_tenant(client):
    body = client.get("/api/status", headers=ACME_AUTH).json()
    assert body["tenant_id"] == "acme"
    assert body["tenant_name"] == "Acme University"
    assert body["multi_tenant"] is True

    body = client.get("/api/status", headers=GLOBEX_AUTH).json()
    assert body["tenant_id"] == "globex"


# ------------------------------------------------------- document reads

def test_document_lists_are_per_tenant(loaded):
    acme = loaded.get("/api/documents", headers=ACME_AUTH).json()
    globex = loaded.get("/api/documents", headers=GLOBEX_AUTH).json()

    assert [d["name"] for d in acme] == ["cs1428_syllabus.md"]
    assert [d["name"] for d in globex] == ["returns_and_shipping.md"]


def test_status_counts_are_per_tenant(loaded):
    acme = loaded.get("/api/status", headers=ACME_AUTH).json()
    globex = loaded.get("/api/status", headers=GLOBEX_AUTH).json()

    assert acme["documents"] == 1 and globex["documents"] == 1
    # Different documents chunk differently; equal counts would hint at a share.
    assert acme["chunks"] > 0 and globex["chunks"] > 0


# -------------------------------------------------- cross-tenant access

def test_a_tenant_cannot_read_another_tenants_document_by_id(loaded):
    """The horizontal-privilege case: a real, valid id from the wrong tenant."""
    acme_doc = loaded.get("/api/documents", headers=ACME_AUTH).json()[0]

    response = loaded.put(
        "/api/documents/{}/metadata".format(acme_doc["doc_id"]),
        json={"metadata": {"hijacked": "yes"}},
        headers=GLOBEX_AUTH,
    )
    assert response.status_code == 404, (
        "Globex reached Acme's document via its id (status {})".format(
            response.status_code
        )
    )

    # And Acme's metadata is untouched.
    still = loaded.get("/api/documents", headers=ACME_AUTH).json()[0]
    assert "hijacked" not in still["metadata"]


def test_a_tenant_cannot_delete_another_tenants_document(loaded):
    acme_doc = loaded.get("/api/documents", headers=ACME_AUTH).json()[0]

    response = loaded.delete(
        "/api/documents/{}".format(acme_doc["doc_id"]), headers=GLOBEX_AUTH
    )
    assert response.status_code == 404

    assert len(loaded.get("/api/documents", headers=ACME_AUTH).json()) == 1


def test_clearing_one_tenant_leaves_the_other_intact(loaded):
    assert loaded.delete("/api/documents", headers=GLOBEX_AUTH).status_code == 200

    assert loaded.get("/api/documents", headers=GLOBEX_AUTH).json() == []
    assert len(loaded.get("/api/documents", headers=ACME_AUTH).json()) == 1


def test_search_cannot_retrieve_another_tenants_content(loaded):
    """A question only the *other* tenant's corpus can answer."""
    response = loaded.post(
        "/api/search",
        json={"query": "What happens if I miss an exam?", "top_k": 10},
        headers=GLOBEX_AUTH,
    )
    assert response.status_code == 200
    names = {hit["doc_name"] for hit in response.json()["hits"]}
    assert names <= {"returns_and_shipping.md"}, (
        "Globex retrieved Acme's documents: {}".format(names)
    )


def test_chat_refuses_a_question_only_the_other_tenant_can_answer(loaded):
    """Out of scope for Globex *because* the syllabus is not its corpus.

    This refusal is produced without an LLM call, so it is also proof that the
    other tenant's content never entered the prompt.
    """
    response = loaded.post(
        "/api/chat",
        json={"message": "What happens if I miss an exam?"},
        headers=GLOBEX_AUTH,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["in_scope"] is False
    assert "cs1428" not in json.dumps(body).lower()


def test_a_refusal_does_not_name_another_tenants_documents(loaded):
    """Refusals list the corpus to be helpful — only the caller's own."""
    response = loaded.post(
        "/api/chat",
        json={"message": "What is the airspeed velocity of a swallow?"},
        headers=GLOBEX_AUTH,
    )
    body = response.json()
    assert "syllabus" not in body["answer"].lower()
    assert "cs1428" not in body["answer"].lower()


def test_facets_do_not_leak_another_tenants_tags(client, syllabus_text, support_text):
    upload(client, "a.md", syllabus_text, ACME_AUTH, metadata={"course": "CS1428"})
    upload(client, "b.md", support_text, GLOBEX_AUTH, metadata={"brand": "Northwind"})

    acme = client.get("/api/facets", headers=ACME_AUTH).json()["facets"]
    globex = client.get("/api/facets", headers=GLOBEX_AUTH).json()["facets"]

    assert "course" in acme and "brand" not in acme
    assert "brand" in globex and "course" not in globex


def test_personas_are_isolated(client):
    client.put(
        "/api/persona",
        json={"name": "Acme TA", "topic": "the syllabus", "style": "", "greeting": ""},
        headers=ACME_AUTH,
    )
    client.put(
        "/api/persona",
        json={"name": "Globex Support", "topic": "returns", "style": "", "greeting": ""},
        headers=GLOBEX_AUTH,
    )

    assert client.get("/api/persona", headers=ACME_AUTH).json()["name"] == "Acme TA"
    assert (
        client.get("/api/persona", headers=GLOBEX_AUTH).json()["name"]
        == "Globex Support"
    )


def test_a_tenant_id_in_the_body_does_not_change_whose_data_is_used(loaded):
    """Identity must come from the credential, never from the request.

    A body field named ``tenant_id`` is ignored. If it were honoured, anyone
    could read any corpus by naming it.
    """
    response = loaded.post(
        "/api/search",
        json={"query": "exam", "top_k": 10, "tenant_id": "acme"},
        headers=GLOBEX_AUTH,
    )
    assert response.status_code in (200, 422)
    if response.status_code == 200:
        names = {hit["doc_name"] for hit in response.json()["hits"]}
        assert names <= {"returns_and_shipping.md"}


# --------------------------------------------- single-tenant compatibility

def test_single_tenant_mode_needs_no_key(single_tenant_client, syllabus_text):
    """The default mode must keep working exactly as before."""
    assert upload(
        single_tenant_client, "cs1428_syllabus.md", syllabus_text, auth={}
    ).status_code == 200

    response = single_tenant_client.get("/api/documents")
    assert response.status_code == 200
    assert [d["name"] for d in response.json()] == ["cs1428_syllabus.md"]


def test_single_tenant_status_reports_the_mode(single_tenant_client):
    body = single_tenant_client.get("/api/status").json()
    assert body["multi_tenant"] is False
    assert body["tenant_id"] == "default"
