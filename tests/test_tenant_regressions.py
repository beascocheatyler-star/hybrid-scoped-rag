"""Regressions for defects found by auditing the multi-tenancy implementation.

Each test here corresponds to a bug that was present in commit 735e107 and is
written to fail against that code. They are kept separate from the main tenancy
suite because their value is historical: they pin specific mistakes so a future
refactor cannot quietly reintroduce them.

The first one is the serious one. The others range from a silent corpus merge
to a crash that turned a 401 into a 500.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import tempfile
import threading

import pytest
from fastapi.testclient import TestClient

from app.core.persona import PRESETS, Persona, PersonaStore
from app.core.tenancy import (
    InvalidTenantError,
    TenantRegistry,
    UnknownTenantError,
)

ACME_KEY = "acme-secret-key-1234"


# ------------------------------------------------- the authentication bypass

def test_the_single_tenant_sentinel_is_not_a_usable_credential():
    """The default tenant must be unreachable by any API key.

    The original code registered it with the literal key
    ``"__single_tenant__"``. That string is public in the source, so in
    multi-tenant mode anyone could send it as X-API-Key and get full
    read/write/delete on the default tenant's corpus.
    """
    with tempfile.TemporaryDirectory() as d:
        registry = TenantRegistry(data_dir=pathlib.Path(d))
        default = registry.register_keyless("default", "Default")
        registry.register("acme", "Acme", ACME_KEY)

        for guess in ["__single_tenant__", "default", "", "   ", None]:
            with pytest.raises(UnknownTenantError):
                registry.resolve(guess)

        # Still usable internally, which is how single-tenant mode works.
        assert registry.context_for(default) is not None
        assert registry.resolve(ACME_KEY).tenant_id == "acme"


def test_sentinel_key_is_rejected_over_http(tmp_path, monkeypatch):
    """End-to-end: the bypass must not work through the API either."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MULTI_TENANT", "1")
    monkeypatch.setenv("TENANTS", "acme:Acme:{}".format(ACME_KEY))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    import app.config as config
    import app.core.embedding as embedding

    config.reset_settings()
    embedding.reset_embedder()
    import app.main

    importlib.reload(app.main)

    with TestClient(app.main.app) as client:
        for guess in ["__single_tenant__", "default"]:
            response = client.get("/api/documents", headers={"X-API-Key": guess})
            assert response.status_code == 401, (
                "{!r} authenticated as a tenant".format(guess)
            )
        assert (
            client.get("/api/documents", headers={"X-API-Key": ACME_KEY}).status_code
            == 200
        )

    config.reset_settings()
    embedding.reset_embedder()


def test_a_configured_tenant_may_not_collide_with_the_default(tmp_path, monkeypatch):
    """Startup must fail loudly rather than share the default's corpus."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MULTI_TENANT", "1")
    monkeypatch.setenv("DEFAULT_TENANT_ID", "acme")
    monkeypatch.setenv("TENANTS", "acme:Acme:{}".format(ACME_KEY))

    import app.config as config
    import app.core.embedding as embedding

    config.reset_settings()
    embedding.reset_embedder()
    import app.main

    # RuntimeError, not InvalidTenantError: this runs at import, before the
    # exception handlers exist, so it is re-raised with a message naming the
    # setting to fix rather than surfacing as a bare traceback.
    with pytest.raises(RuntimeError, match="DEFAULT_TENANT_ID"):
        importlib.reload(app.main)

    config.reset_settings()
    embedding.reset_embedder()
    monkeypatch.delenv("DEFAULT_TENANT_ID")
    monkeypatch.delenv("TENANTS")
    importlib.reload(app.main)


# ----------------------------------------------------- the silent corpus merge

def test_tenant_ids_differing_only_by_case_are_rejected():
    """``ACME`` and ``acme`` normalise to one id and must not both register.

    Previously both were accepted, both resolved, and both got the *same*
    DocumentStore — so one customer could read and delete the other's
    documents, with no error and nothing in the log to notice.
    """
    with tempfile.TemporaryDirectory() as d:
        registry = TenantRegistry(data_dir=pathlib.Path(d))
        registry.register("ACME", "Acme Corp", "key-upper")

        with pytest.raises(InvalidTenantError):
            registry.register("acme", "Acme Rival", "key-lower")

        # The rejected tenant's key must not work.
        with pytest.raises(UnknownTenantError):
            registry.resolve("key-lower")


def test_re_registering_an_id_is_an_error_not_an_overwrite():
    with tempfile.TemporaryDirectory() as d:
        registry = TenantRegistry(data_dir=pathlib.Path(d))
        registry.register("acme", "Acme", "key-one")
        with pytest.raises(InvalidTenantError):
            registry.register("acme", "Acme Again", "key-two")

        assert registry.resolve("key-one").name == "Acme"


def test_a_key_cannot_be_shared_by_two_tenants():
    with tempfile.TemporaryDirectory() as d:
        registry = TenantRegistry(data_dir=pathlib.Path(d))
        registry.register("acme", "Acme", ACME_KEY)
        with pytest.raises(InvalidTenantError):
            registry.register("globex", "Globex", ACME_KEY)


# -------------------------------------------------------- the 500-not-401 bug

@pytest.mark.parametrize("key", ["kéy", "日本語", "kéy-with-ascii", "🔑"])
def test_non_ascii_keys_are_rejected_cleanly(key):
    """``hmac.compare_digest`` raises TypeError on non-ASCII ``str``.

    That turned any UTF-8 X-API-Key into an unhandled 500 with the presented
    key in the traceback, instead of the uniform 401 the module goes to some
    length to guarantee.
    """
    with tempfile.TemporaryDirectory() as d:
        registry = TenantRegistry(data_dir=pathlib.Path(d))
        registry.register("acme", "Acme", ACME_KEY)

        with pytest.raises(UnknownTenantError):
            registry.resolve(key)


def test_non_ascii_key_returns_401_over_http(tmp_path, monkeypatch):
    """A header carrying non-ASCII bytes must 401, not 500.

    HTTP headers are ASCII by spec and the client refuses to encode a raw
    ``é``, so the byte sequence is sent explicitly. Starlette decodes headers
    as latin-1, which turns those bytes into a non-ASCII ``str`` — exactly the
    input that made ``hmac.compare_digest`` raise.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MULTI_TENANT", "1")
    monkeypatch.setenv("TENANTS", "acme:Acme:{}".format(ACME_KEY))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    import app.config as config
    import app.core.embedding as embedding

    config.reset_settings()
    embedding.reset_embedder()
    import app.main

    importlib.reload(app.main)

    with TestClient(app.main.app) as client:
        response = client.get(
            "/api/documents", headers={"X-API-Key": "k\xe9y".encode("latin-1")}
        )
        assert response.status_code == 401, (
            "non-ASCII key produced {} instead of 401".format(response.status_code)
        )

    config.reset_settings()
    embedding.reset_embedder()


# ------------------------------------------------------- the shared persona

def test_two_tenants_applying_one_preset_get_separate_objects():
    """``PRESETS`` holds mutable Personas; storing them by reference aliased
    every tenant that applied the same preset onto one object."""
    with tempfile.TemporaryDirectory() as d:
        a, b = pathlib.Path(d) / "a", pathlib.Path(d) / "b"
        a.mkdir()
        b.mkdir()
        pa, pb = PersonaStore(a), PersonaStore(b)

        pa.apply_preset("course_ta")
        pb.apply_preset("course_ta")

        assert pa.current is not pb.current

        pa.current.suggested_questions.append("TENANT A CONFIDENTIAL")
        assert "TENANT A CONFIDENTIAL" not in pb.current.suggested_questions


def test_applying_a_preset_does_not_mutate_the_global_presets():
    """Pollution here would persist for the life of the process and reach
    every later tenant, plus the unauthenticated /api/persona/presets."""
    before = list(PRESETS["customer_care"].suggested_questions)

    with tempfile.TemporaryDirectory() as d:
        store = PersonaStore(pathlib.Path(d))
        store.apply_preset("customer_care")
        store.current.suggested_questions.append("LEAKED")
        store.current.name = "Mutated"

    assert PRESETS["customer_care"].suggested_questions == before
    assert PRESETS["customer_care"].name != "Mutated"


# --------------------------------------------------- persona write durability

def test_persona_write_is_atomic_and_leaves_no_debris():
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d)
        store = PersonaStore(path)
        store.set(Persona(name="Acme TA", topic="the syllabus"))

        saved = json.loads((path / "persona.json").read_text("utf-8"))
        assert saved["name"] == "Acme TA"
        assert [p for p in path.iterdir() if ".tmp" in p.name] == []


def test_a_failed_persona_write_changes_nothing(monkeypatch):
    """The in-memory persona was previously adopted *before* the write, so a
    failed write left the running bot using a persona that was never saved."""
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d)
        store = PersonaStore(path)
        store.set(Persona(name="Original", topic="t"))

        def boom(_fd):
            raise OSError("disk full")

        monkeypatch.setattr(os, "fsync", boom)
        with pytest.raises(OSError):
            store.set(Persona(name="Should Not Stick", topic="t"))
        monkeypatch.undo()

        assert store.current.name == "Original"
        assert json.loads((path / "persona.json").read_text("utf-8"))["name"] == "Original"
        assert [p for p in path.iterdir() if ".tmp" in p.name] == []


def test_concurrent_persona_writes_never_produce_invalid_json():
    """``write_text`` truncates before writing, so interleaved writes could
    leave JSON that fails to parse — silently resetting the tenant to defaults
    at the next boot."""
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d)
        store = PersonaStore(path)
        errors = []

        def write(i):
            try:
                store.set(Persona(name="Tenant {}".format(i), topic="t" * (i % 50 + 1)))
            except Exception as exc:  # pragma: no cover - would be a failure
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        body = json.loads((path / "persona.json").read_text("utf-8"))
        assert body["name"].startswith("Tenant")
        assert [p for p in path.iterdir() if ".tmp" in p.name] == []


# ------------------------------------------------------------- concurrency

def test_concurrent_first_requests_build_one_context_per_tenant():
    """``context_for`` double-checks under a per-tenant lock. Two threads
    racing on a cold tenant must not build two stores over one directory."""
    with tempfile.TemporaryDirectory() as d:
        registry = TenantRegistry(data_dir=pathlib.Path(d))
        tenant = registry.register("acme", "Acme", ACME_KEY)

        seen = []
        barrier = threading.Barrier(8)

        def grab():
            barrier.wait()
            seen.append(registry.context_for(tenant))

        threads = [threading.Thread(target=grab) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(seen) == 8
        assert all(ctx is seen[0] for ctx in seen), "built more than one store"


def test_context_for_rejects_an_unregistered_tenant():
    """A Tenant object is not itself authority to open a corpus."""
    from app.core.tenancy import Tenant

    with tempfile.TemporaryDirectory() as d:
        registry = TenantRegistry(data_dir=pathlib.Path(d))
        registry.register("acme", "Acme", ACME_KEY)

        forged = Tenant(tenant_id="globex", name="Globex")
        with pytest.raises(UnknownTenantError):
            registry.context_for(forged)

        # Even an id that *is* registered, but a different object than the one
        # the registry issued, is refused.
        impostor = Tenant(tenant_id="acme", name="Not Acme")
        with pytest.raises(UnknownTenantError):
            registry.context_for(impostor)


def test_an_invalid_default_tenant_id_fails_with_a_readable_error(
    tmp_path, monkeypatch
):
    """Startup must name the setting to fix, not emit a bare traceback."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEFAULT_TENANT_ID", "Not A Valid Id!")

    import app.config as config
    import app.core.embedding as embedding

    config.reset_settings()
    embedding.reset_embedder()
    import app.main

    with pytest.raises(RuntimeError, match="DEFAULT_TENANT_ID"):
        importlib.reload(app.main)

    config.reset_settings()
    embedding.reset_embedder()
    monkeypatch.delenv("DEFAULT_TENANT_ID")
    importlib.reload(app.main)


def test_a_malformed_tenants_string_fails_with_a_readable_error(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TENANTS", "this-entry-has-no-colons")

    import app.config as config
    import app.core.embedding as embedding

    config.reset_settings()
    embedding.reset_embedder()
    import app.main

    with pytest.raises(RuntimeError, match="TENANTS"):
        importlib.reload(app.main)

    config.reset_settings()
    embedding.reset_embedder()
    monkeypatch.delenv("TENANTS")
    importlib.reload(app.main)


def test_persona_fsync_is_not_serialised_under_the_lock():
    """The lock must not be held across the slow part of the write.

    Holding it through fsync serialises concurrent persona updates, and since
    the endpoints are sync `def` each blocked request occupies a threadpool
    worker — a burst could stall unrelated endpoints.
    """
    import app.core.persona as persona_module

    with tempfile.TemporaryDirectory() as d:
        store = PersonaStore(pathlib.Path(d))

        in_fsync = threading.Event()
        proceed = threading.Event()
        real_fsync = os.fsync
        calls = []

        def gated_fsync(fd):
            calls.append(fd)
            if len(calls) == 1:
                in_fsync.set()
                proceed.wait(timeout=5)
            return real_fsync(fd)

        persona_module.os.fsync = gated_fsync
        try:
            slow = threading.Thread(
                target=lambda: store.set(Persona(name="Slow", topic="t"))
            )
            slow.start()
            assert in_fsync.wait(timeout=5), "first write never reached fsync"

            # Reading the current persona must not block behind that fsync.
            reader = threading.Thread(target=lambda: store.current)
            reader.start()
            reader.join(timeout=2)
            assert not reader.is_alive(), "reading persona blocked on a write fsync"

            proceed.set()
            slow.join(timeout=5)
            assert not slow.is_alive()
        finally:
            persona_module.os.fsync = real_fsync

        assert store.current.name == "Slow"


def test_one_tenants_cold_start_does_not_block_another(monkeypatch):
    """The registry lock must not be held across store construction.

    Building a store reads the corpus and may re-embed every chunk. Holding the
    global lock across that made one tenant's cold start stall every other
    tenant's first request.
    """
    import app.core.tenancy as tenancy

    with tempfile.TemporaryDirectory() as d:
        registry = TenantRegistry(data_dir=pathlib.Path(d))
        slow = registry.register("slow", "Slow", "key-slow")
        fast = registry.register("fast", "Fast", "key-fast")

        started = threading.Event()
        release = threading.Event()
        real_store = tenancy.DocumentStore

        class GatedStore(real_store):  # type: ignore[valid-type,misc]
            def __init__(self, *args, **kwargs):
                if "slow" in str(kwargs.get("data_dir", "")):
                    started.set()
                    # Block until the other tenant has finished entirely.
                    assert release.wait(timeout=5), "fast tenant never completed"
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(tenancy, "DocumentStore", GatedStore)

        done = []
        slow_thread = threading.Thread(target=lambda: registry.context_for(slow))
        slow_thread.start()

        assert started.wait(timeout=5), "slow tenant never began building"

        # With a global lock held across construction, this would deadlock
        # until the timeout above fires.
        fast_ctx = registry.context_for(fast)
        done.append(fast_ctx)
        release.set()
        slow_thread.join(timeout=5)

        assert not slow_thread.is_alive()
        assert done and done[0].tenant_id == "fast"
