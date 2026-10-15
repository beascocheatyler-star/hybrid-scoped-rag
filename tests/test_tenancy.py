"""Tenant isolation: the property that one customer can never read another's.

Written adversarially on purpose. A test that only checks the happy path —
"tenant A uploads, tenant A retrieves" — passes just as happily on a system with
no isolation whatsoever, because a single shared corpus also returns A's
document to A. Every test here instead tries to *reach across* the boundary and
asserts that it fails.

Two tenants with unmistakably different corpora make a leak impossible to
mistake for a near miss: a course syllabus and a retail returns policy share
almost no vocabulary, so a hit from the wrong one cannot be explained away as a
borderline relevance call.
"""

from __future__ import annotations

import pytest

from app.core.tenancy import (
    InvalidTenantError,
    TenantRegistry,
    UnknownTenantError,
    validate_tenant_id,
)

ACME_KEY = "key-acme-0000000000"
GLOBEX_KEY = "key-globex-111111111"


@pytest.fixture
def registry(tmp_path):
    reg = TenantRegistry(
        data_dir=tmp_path,
        store_options={"chunk_size": 1200, "overlap": 200},
    )
    reg.register("acme", "Acme University", ACME_KEY)
    reg.register("globex", "Globex Retail", GLOBEX_KEY)
    return reg


@pytest.fixture
def loaded(registry, syllabus_text, support_text):
    """Acme holds the syllabus; Globex holds the returns policy."""
    acme = registry.context_for(registry.resolve(ACME_KEY))
    globex = registry.context_for(registry.resolve(GLOBEX_KEY))
    acme.store.add("cs1428_syllabus.md", syllabus_text.encode("utf-8"))
    globex.store.add("returns_and_shipping.md", support_text.encode("utf-8"))
    return registry, acme, globex


# ------------------------------------------------------- id validation

@pytest.mark.parametrize(
    "bad_id",
    [
        "../globex",          # classic traversal
        "..",
        "acme/../globex",     # traversal that ends somewhere real
        "acme/docs",          # separator
        "acme\\docs",         # separator, Windows flavour
        "",
        "   ",
        ".",
        "-acme",              # must start alphanumeric
        "_acme",
        "a" * 64,             # over the length cap
        "acme\x00",           # null byte truncation attack
        "acme\n",
        "tenants",            # reserved
        "shared",
    ],
)
def test_unsafe_tenant_ids_are_rejected(bad_id):
    with pytest.raises(InvalidTenantError):
        validate_tenant_id(bad_id)


@pytest.mark.parametrize(
    "good_id,expected",
    [("acme", "acme"), ("ACME", "acme"), ("acme-2", "acme-2"), ("a1_b2", "a1_b2")],
)
def test_valid_tenant_ids_are_accepted_and_normalised(good_id, expected):
    assert validate_tenant_id(good_id) == expected


def test_traversal_cannot_escape_the_tenant_root(registry, tmp_path):
    """A traversal id must not resolve to a path outside the tenant root."""
    for attempt in ["../escape", "..", "acme/../../etc"]:
        with pytest.raises(InvalidTenantError):
            registry.directory_for(attempt)


# ------------------------------------------------------------ isolation

def test_each_tenant_gets_its_own_directory(registry):
    acme = registry.directory_for("acme")
    globex = registry.directory_for("globex")
    assert acme != globex
    assert acme.parent == globex.parent


def test_a_document_is_visible_only_to_its_owner(loaded):
    _registry, acme, globex = loaded

    assert [d.name for d in acme.store.list_documents()] == ["cs1428_syllabus.md"]
    assert [d.name for d in globex.store.list_documents()] == [
        "returns_and_shipping.md"
    ]


def test_retrieval_cannot_reach_another_tenants_corpus(loaded):
    """The decisive test: a question answerable *only* by the other tenant.

    Globex's corpus says nothing about exams. If tenant isolation were merely a
    post-retrieval filter, the syllabus chunks would still be ranked here and a
    filtering bug would surface them. They are not in this index at all.
    """
    _registry, acme, globex = loaded

    hits = globex.store.index.search("What happens if I miss an exam?", top_k=10)
    assert all(h.chunk.doc_name == "returns_and_shipping.md" for h in hits), (
        "Globex retrieved a document it does not own: {}".format(
            sorted({h.chunk.doc_name for h in hits})
        )
    )

    hits = acme.store.index.search("How long do I have to return an item?", top_k=10)
    assert all(h.chunk.doc_name == "cs1428_syllabus.md" for h in hits)


def test_indexes_are_not_shared_objects(loaded):
    _registry, acme, globex = loaded
    assert acme.store is not globex.store
    assert acme.store.index is not globex.store.index
    assert acme.store.total_chunks > 0
    assert globex.store.total_chunks > 0


def test_deleting_one_tenants_document_leaves_the_other_intact(loaded):
    _registry, acme, globex = loaded

    doc_id = acme.store.list_documents()[0].doc_id
    assert acme.store.delete(doc_id) is True

    assert acme.store.is_empty
    assert not globex.store.is_empty, "Deleting Acme's document emptied Globex"


def test_clearing_one_tenant_does_not_clear_another(loaded):
    _registry, acme, globex = loaded

    acme.store.clear()

    assert acme.store.is_empty
    assert len(globex.store.list_documents()) == 1


def test_the_same_file_may_be_uploaded_by_two_tenants(registry, syllabus_text):
    """Deduplication is per-tenant.

    The duplicate check exists to stop one customer uploading the same file
    twice. Applying it across tenants would both leak information — B learns A
    holds an identical file — and wrongly deny B its own copy.
    """
    acme = registry.context_for(registry.resolve(ACME_KEY))
    globex = registry.context_for(registry.resolve(GLOBEX_KEY))

    data = syllabus_text.encode("utf-8")
    acme.store.add("shared.md", data)
    globex.store.add("shared.md", data)  # must not raise

    assert len(acme.store.list_documents()) == 1
    assert len(globex.store.list_documents()) == 1


def test_personas_are_per_tenant(loaded):
    from app.core.persona import Persona

    _registry, acme, globex = loaded

    acme.personas.set(Persona(name="Acme TA", topic="the CS 1428 syllabus"))
    globex.personas.set(Persona(name="Globex Support", topic="returns and shipping"))

    assert acme.personas.current.name == "Acme TA"
    assert globex.personas.current.name == "Globex Support"


# ----------------------------------------------------------- credentials

def test_an_unknown_key_resolves_to_no_tenant(registry):
    for bad in ["key-nonexistent", "", None, "   "]:
        with pytest.raises(UnknownTenantError):
            registry.resolve(bad)


def test_a_key_resolves_only_to_its_own_tenant(registry):
    assert registry.resolve(ACME_KEY).tenant_id == "acme"
    assert registry.resolve(GLOBEX_KEY).tenant_id == "globex"


def test_a_key_cannot_be_rebound_to_another_tenant(registry):
    """Rebinding a live key would silently redirect one tenant's traffic."""
    with pytest.raises(InvalidTenantError):
        registry.register("globex", "Globex Retail", ACME_KEY)

    assert registry.resolve(ACME_KEY).tenant_id == "acme"


def test_registering_requires_a_key(registry):
    for bad in ["", "   ", None]:
        with pytest.raises(InvalidTenantError):
            registry.register("initech", "Initech", bad)


def test_context_is_cached_per_tenant(registry):
    """Rebuilding a store per request would drop the embedding cache."""
    tenant = registry.resolve(ACME_KEY)
    assert registry.context_for(tenant) is registry.context_for(tenant)


def test_tenant_state_survives_a_registry_restart(tmp_path, syllabus_text):
    """Isolation is on disk, not just in memory."""
    first = TenantRegistry(data_dir=tmp_path, store_options={"chunk_size": 1200})
    first.register("acme", "Acme", ACME_KEY)
    ctx = first.context_for(first.resolve(ACME_KEY))
    ctx.store.add("cs1428_syllabus.md", syllabus_text.encode("utf-8"))

    second = TenantRegistry(data_dir=tmp_path, store_options={"chunk_size": 1200})
    second.register("acme", "Acme", ACME_KEY)
    reloaded = second.context_for(second.resolve(ACME_KEY))

    assert [d.name for d in reloaded.store.list_documents()] == ["cs1428_syllabus.md"]

    # And a tenant that never uploaded anything still starts empty.
    second.register("globex", "Globex", GLOBEX_KEY)
    assert second.context_for(second.resolve(GLOBEX_KEY)).store.is_empty
