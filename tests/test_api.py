"""End-to-end HTTP tests.

The LLM is stubbed, so the whole suite runs offline with no API key and no
spend. What is verified here is the engine's own logic: routing, scope
enforcement, citation assembly, and the prompt actually sent upstream.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SYLLABUS = EXAMPLES / "course_ta" / "cs1428_syllabus.md"


class StubBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class StubUsage:
    input_tokens = 1234
    output_tokens = 56


class StubResponse:
    stop_reason = "end_turn"
    model = "stub-model"

    def __init__(self, text: str):
        self.content = [StubBlock(text)]
        self.usage = StubUsage()


class StubMessages:
    """Records the last request so tests can assert on the prompt."""

    def __init__(self, reply: str = "A grounded answer."):
        self.reply = reply
        self.last_kwargs = None
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        self.last_kwargs = kwargs
        return StubResponse(self.reply)


class StubClient:
    def __init__(self, reply: str = "A grounded answer."):
        self.messages = StubMessages(reply)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient wired to a throwaway data dir and a stubbed LLM."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    import app.config as config

    config.reset_settings()

    # Reimport main so module-level singletons bind to the temp directory.
    import importlib

    import app.main as main

    importlib.reload(main)

    stub = StubClient()
    main.engine._client = stub

    test_client = TestClient(main.app)
    test_client.stub = stub  # type: ignore[attr-defined]
    test_client.main = main  # type: ignore[attr-defined]
    yield test_client

    config.reset_settings()


def upload_syllabus(client):
    return client.post(
        "/api/documents",
        files={"files": ("cs1428_syllabus.md", SYLLABUS.read_bytes(), "text/markdown")},
    )


# ---------------------------------------------------------------- status

def test_status_reports_configuration(client):
    body = client.get("/api/status").json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is True
    assert body["documents"] == 0
    assert "greeting" in body


def test_supported_types_lists_extensions(client):
    extensions = client.get("/api/supported-types").json()["extensions"]
    assert ".pdf" in extensions
    assert ".md" in extensions


# ------------------------------------------------------------- documents

def test_upload_indexes_a_document(client):
    response = upload_syllabus(client)
    assert response.status_code == 200
    body = response.json()
    assert len(body["uploaded"]) == 1
    assert body["total_chunks"] > 0
    assert body["uploaded"][0]["name"] == "cs1428_syllabus.md"


def test_upload_rejects_an_unsupported_type(client):
    response = client.post(
        "/api/documents",
        files={"files": ("photo.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_reports_partial_failure(client):
    """One bad file must not discard the good ones."""
    response = client.post(
        "/api/documents",
        files=[
            ("files", ("good.md", b"The final exam is cumulative.", "text/markdown")),
            ("files", ("bad.png", b"\x89PNG", "image/png")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["uploaded"]) == 1
    assert len(body["skipped"]) == 1
    assert body["skipped"][0]["name"] == "bad.png"


def test_duplicate_upload_is_skipped(client):
    upload_syllabus(client)
    response = upload_syllabus(client)
    assert response.status_code == 400
    assert "already in the knowledge base" in response.json()["detail"]


def test_list_and_delete_documents(client):
    doc_id = upload_syllabus(client).json()["uploaded"][0]["doc_id"]

    assert len(client.get("/api/documents").json()) == 1

    assert client.delete("/api/documents/{}".format(doc_id)).status_code == 200
    assert client.get("/api/documents").json() == []


def test_delete_unknown_document_is_404(client):
    assert client.delete("/api/documents/nope").status_code == 404


def test_clear_documents(client):
    upload_syllabus(client)
    assert client.delete("/api/documents").json()["total_documents"] == 0
    assert client.get("/api/documents").json() == []


# --------------------------------------------------------------- persona

def test_apply_preset(client):
    body = client.post("/api/persona/preset", json={"preset": "course_ta"}).json()
    assert "CS 1428" in body["name"]
    assert body["suggested_questions"]


def test_unknown_preset_is_rejected(client):
    response = client.post("/api/persona/preset", json={"preset": "nonsense"})
    assert response.status_code == 400


def test_custom_persona_round_trips(client):
    payload = {
        "name": "the Library Assistant",
        "topic": "library hours and borrowing rules",
        "style": "Be brief.",
        "greeting": "",
        "suggested_questions": [],
    }
    assert client.put("/api/persona", json=payload).json()["name"] == payload["name"]
    assert client.get("/api/persona").json()["topic"] == payload["topic"]


def test_persona_survives_reload(client):
    client.post("/api/persona/preset", json={"preset": "customer_care"})
    assert "Northwind" in client.get("/api/status").json()["persona"]["name"]


# ------------------------------------------------------------------ chat

def test_chat_without_documents_reports_no_corpus(client):
    body = client.post("/api/chat", json={"message": "Anything?"}).json()
    assert body["decision"] == "no_corpus"
    assert body["in_scope"] is False
    assert client.stub.messages.call_count == 0, "Must not call the LLM"


def test_in_scope_question_is_answered_with_citations(client):
    upload_syllabus(client)
    body = client.post(
        "/api/chat", json={"message": "How is the final grade calculated?"}
    ).json()

    assert body["in_scope"] is True
    assert body["answer"] == "A grounded answer."
    assert body["citations"], "An answered question must carry sources"
    assert body["citations"][0]["doc_name"] == "cs1428_syllabus.md"
    assert body["usage"]["input_tokens"] == 1234
    assert client.stub.messages.call_count == 1


def test_off_topic_question_is_refused_without_an_llm_call(client):
    upload_syllabus(client)
    body = client.post(
        "/api/chat", json={"message": "What is the capital of France?"}
    ).json()

    assert body["in_scope"] is False
    assert body["decision"] == "out_of_scope"
    assert body["citations"] == []
    assert client.stub.messages.call_count == 0, (
        "An off-topic question must be refused before any token is spent"
    )


def test_refusal_names_the_scope_and_documents(client):
    upload_syllabus(client)
    client.post("/api/persona/preset", json={"preset": "course_ta"})
    body = client.post(
        "/api/chat", json={"message": "Give me a recipe for lasagna."}
    ).json()
    assert "CS 1428" in body["answer"]
    assert "cs1428_syllabus.md" in body["answer"]


def test_empty_message_is_rejected(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_history_is_replayed_to_the_model(client):
    upload_syllabus(client)
    client.post(
        "/api/chat",
        json={
            "message": "And what about the final?",
            "history": [
                {"role": "user", "content": "When is exam 1?"},
                {"role": "assistant", "content": "February 18."},
            ],
        },
    )
    messages = client.stub.messages.last_kwargs["messages"]
    assert len(messages) == 3
    assert messages[0]["content"] == "When is exam 1?"
    assert messages[-1]["role"] == "user"


def test_history_is_trimmed_to_start_with_a_user_turn(client):
    """The API requires messages[0] to be a user turn."""
    upload_syllabus(client)
    client.post(
        "/api/chat",
        json={
            "message": "How are grades calculated?",
            "history": [{"role": "assistant", "content": "Orphaned reply."}],
        },
    )
    messages = client.stub.messages.last_kwargs["messages"]
    assert messages[0]["role"] == "user"


def test_messages_never_have_consecutive_user_turns(client):
    """A trailing user turn would merge with the grounded turn.

    The API combines consecutive same-role messages, which would splice a stale
    question into the same turn as the fresh CONTEXT block.
    """
    upload_syllabus(client)
    client.post(
        "/api/chat",
        json={
            "message": "What about the final exam?",
            "history": [
                {"role": "user", "content": "When is exam 1?"},
                {"role": "assistant", "content": "February 18."},
                {"role": "user", "content": "A question with no reply yet."},
            ],
        },
    )
    messages = client.stub.messages.last_kwargs["messages"]
    roles = [m["role"] for m in messages]

    assert "A question with no reply yet." not in [m["content"] for m in messages]
    for earlier, later in zip(roles, roles[1:]):
        assert earlier != later, "Consecutive {} turns in {}".format(earlier, roles)
    assert roles[0] == "user"
    assert roles[-1] == "user"


def test_history_roles_are_validated(client):
    """An unknown role should be a visible 422, not a silently dropped turn."""
    upload_syllabus(client)
    response = client.post(
        "/api/chat",
        json={
            "message": "How are grades calculated?",
            "history": [{"role": "system", "content": "sneaky"}],
        },
    )
    assert response.status_code == 422


def test_prompt_contains_grounding_and_security_rules(client):
    upload_syllabus(client)
    client.post("/api/chat", json={"message": "What is the late work policy?"})

    system = client.stub.messages.last_kwargs["system"]
    assert "Answer ONLY from the CONTEXT" in system
    assert "untrusted reference material" in system
    assert "cs1428_syllabus.md" in system

    user_turn = client.stub.messages.last_kwargs["messages"][-1]["content"]
    assert "CONTEXT PASSAGES" in user_turn
    assert "[Source 1]" in user_turn
    assert "What is the late work policy?" in user_turn


def test_retrieved_context_is_actually_relevant(client):
    """Guard against silently shipping empty or wrong context to the model."""
    upload_syllabus(client)
    client.post("/api/chat", json={"message": "What is the late work policy?"})
    user_turn = client.stub.messages.last_kwargs["messages"][-1]["content"]
    assert "late" in user_turn.lower()


def test_small_talk_reaches_the_model(client):
    upload_syllabus(client)
    body = client.post("/api/chat", json={"message": "hello"}).json()
    assert body["decision"] == "small_talk"
    assert client.stub.messages.call_count == 1


def test_doc_ids_filter_restricts_retrieval(client):
    upload_syllabus(client)
    client.post(
        "/api/documents",
        files={"files": ("other.md", b"Unrelated content about gardening.", "text/markdown")},
    )
    doc_id = client.get("/api/documents").json()[0]["doc_id"]

    body = client.post(
        "/api/chat",
        json={"message": "How is the final grade calculated?", "doc_ids": [doc_id]},
    ).json()
    assert all(c["doc_id"] == doc_id for c in body["citations"])


def test_llm_failure_returns_503(client):
    upload_syllabus(client)

    def boom(**kwargs):
        raise RuntimeError("upstream exploded")

    client.stub.messages.create = boom
    response = client.post(
        "/api/chat", json={"message": "How is the final grade calculated?"}
    )
    assert response.status_code == 503
    assert "detail" in response.json()


def test_model_refusal_is_handled(client):
    upload_syllabus(client)

    class RefusingResponse(StubResponse):
        stop_reason = "refusal"

    client.stub.messages.create = lambda **kw: RefusingResponse("")
    body = client.post(
        "/api/chat", json={"message": "How is the final grade calculated?"}
    ).json()
    assert body["in_scope"] is False


# -------------------------------------------------------------------- ui

def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Hybrid Scoped RAG" in response.text


def test_openapi_schema_is_available(client):
    assert client.get("/openapi.json").status_code == 200
