"""Hybrid retrieval: embeddings, RRF fusion, and scoped search."""

from __future__ import annotations

import pytest

from app.core.embedding import (
    HashingEmbedder,
    build_embedder,
    cosine,
    get_embedder,
    reset_embedder,
)
from app.core.metadata import ScopeFilter
from app.core.retrieval import Chunk, HybridIndex


@pytest.fixture
def embedder():
    return HashingEmbedder(dimensions=512)


def make_index(embedder, **kwargs):
    return HybridIndex(embedder=embedder, **kwargs)


def add(index, chunk_id, text, doc_id="d1", metadata=None, position=0):
    index.add_chunks(
        [
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_name="{}.md".format(doc_id),
                position=position,
                text=text,
                metadata=metadata or {},
            )
        ]
    )


# --------------------------------------------------------------- embeddings


def test_embedding_is_deterministic(embedder):
    assert embedder.embed("hello world") == embedder.embed("hello world")


def test_embedding_has_requested_dimensions(embedder):
    assert len(embedder.embed("anything")) == 512


def test_embedding_is_l2_normalized(embedder):
    vector = embedder.embed("the final exam is in week eight")
    assert cosine(vector, vector) == pytest.approx(1.0, abs=1e-9)


def test_empty_text_embeds_to_zero(embedder):
    assert not any(embedder.embed(""))


def test_text_with_no_word_characters_embeds_to_zero(embedder):
    assert not any(embedder.embed("!!! ??? ..."))


def test_similar_text_scores_higher_than_unrelated(embedder):
    base = embedder.embed("the midterm examination schedule")
    similar = embedder.embed("examination schedule for the midterm")
    unrelated = embedder.embed("refund policy for damaged footwear")
    assert cosine(base, similar) > cosine(base, unrelated)


@pytest.mark.parametrize(
    "a,b",
    [
        ("enrolment", "enrollment"),
        ("syllabus", "syllabi"),
        ("organise", "organize"),
        ("returns", "return"),
    ],
)
def test_spelling_and_morphology_variants_are_similar(embedder, a, b):
    """The specific thing character n-grams buy that BM25 cannot do at all.

    BM25 treats these pairs as entirely unrelated tokens — zero overlap. The
    hashing embedder sees shared n-grams, so a question using one spelling can
    still retrieve a passage using the other.
    """
    assert cosine(embedder.embed(a), embedder.embed(b)) > 0.4


def test_cosine_of_mismatched_dimensions_is_zero():
    assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_build_embedder_defaults_to_hashing():
    assert build_embedder("hashing").name == "hashing"


def test_unknown_backend_falls_back_to_hashing():
    """A typo in configuration must not prevent the app from starting."""
    assert build_embedder("not-a-real-backend").name == "hashing"


def test_missing_optional_dependency_falls_back(monkeypatch):
    """sentence-transformers is optional; its absence degrades, not crashes."""
    import app.core.embedding as embedding_module

    def explode(*_args, **_kwargs):
        raise ImportError("sentence_transformers is not installed")

    monkeypatch.setattr(
        embedding_module, "SentenceTransformerEmbedder", explode
    )
    assert build_embedder("sentence-transformers").name == "hashing"


def test_embedder_singleton_is_reused():
    reset_embedder()
    try:
        assert get_embedder() is get_embedder()
    finally:
        reset_embedder()


def test_zero_dimensions_is_rejected():
    with pytest.raises(ValueError):
        HashingEmbedder(dimensions=0)


def test_batch_embedding_matches_individual(embedder):
    texts = ["first passage", "second passage"]
    assert embedder.embed_batch(texts) == [embedder.embed(t) for t in texts]


# ------------------------------------------------------------------- fusion


def test_chunks_are_embedded_on_add(embedder):
    index = make_index(embedder)
    add(index, "c1", "the final exam is in week eight")
    assert index._chunks[0].embedding is not None


def test_hit_records_both_ranks_when_both_retrievers_agree(embedder):
    index = make_index(embedder)
    add(index, "c1", "the final exam is held in week eight of the term")
    add(index, "c2", "refund policy for damaged footwear and apparel", doc_id="d2")

    hit = index.search("when is the final exam")[0]
    assert hit.lexical_rank is not None
    assert hit.dense_rank is not None
    assert set(hit.retrievers) == {"lexical", "dense"}


def test_dense_only_hit_has_no_lexical_rank(embedder):
    """A chunk sharing no query terms can still arrive via the dense retriever."""
    index = make_index(embedder)
    add(index, "c1", "required materials and course supplies")

    hits = index.search("textbook")
    assert hits, "dense retrieval should still surface a candidate"
    assert hits[0].lexical_rank is None
    assert hits[0].dense_rank is not None
    assert hits[0].retrievers == ["dense"]


def test_disabling_dense_weight_yields_pure_lexical(embedder):
    index = make_index(embedder, dense_weight=0.0)
    add(index, "c1", "required materials and course supplies")
    # With no lexical overlap and dense disabled, nothing is retrievable.
    assert index.search("textbook") == []


def test_disabling_lexical_weight_still_retrieves_densely(embedder):
    index = make_index(embedder, lexical_weight=0.0)
    add(index, "c1", "the final exam is in week eight")
    hits = index.search("final exam")
    assert hits
    assert hits[0].dense_rank is not None


def test_rrf_prefers_a_chunk_ranked_well_by_both(embedder):
    """The core claim of fusion: agreement beats a single strong signal."""
    index = make_index(embedder)
    # Mentions the query terms *and* is topically close.
    add(index, "both", "the final exam schedule and exam room for the final")
    # Mentions one term many times but is otherwise a different topic.
    add(index, "lexical_only", "exam exam exam exam exam", doc_id="d2")

    hits = index.search("when is the final exam scheduled")
    assert hits[0].chunk.chunk_id == "both"


def test_rrf_k_affects_ordering(embedder):
    """Expose the constant's effect rather than asserting a magic number."""
    sharp = make_index(embedder, rrf_k=1)
    flat = make_index(embedder, rrf_k=1000)
    for index in (sharp, flat):
        add(index, "c1", "the final exam is in week eight")
        add(index, "c2", "exam policies and exam integrity rules", doc_id="d2")

    assert sharp.search("final exam")[0].score != flat.search("final exam")[0].score


def test_fused_score_is_positive_for_every_hit(embedder):
    index = make_index(embedder)
    add(index, "c1", "the final exam is in week eight")
    assert all(h.score > 0 for h in index.search("final exam"))


def test_negative_similarity_chunks_are_excluded(embedder):
    """A chunk actively dissimilar should not earn RRF credit for placing last."""
    index = make_index(embedder)
    add(index, "c1", "completely unrelated vocabulary about marine biology")
    hits = index.search("quarterly financial reconciliation procedures")
    for hit in hits:
        assert hit.dense_score > 0 or hit.dense_rank is None


# -------------------------------------------------------------------- scope


def test_search_is_restricted_by_metadata(embedder):
    index = make_index(embedder)
    add(index, "c1", "the final exam is in week eight", doc_id="d1",
        metadata={"course": "cs1428"})
    add(index, "c2", "the final exam is in week ten", doc_id="d2",
        metadata={"course": "cs2308"})

    hits = index.search("final exam", scope=ScopeFilter.parse({"course": "cs1428"}))
    assert {h.chunk.doc_id for h in hits} == {"d1"}


def test_scope_matching_nothing_returns_no_hits(embedder):
    index = make_index(embedder)
    add(index, "c1", "the final exam is in week eight", metadata={"course": "cs1428"})
    assert index.search("final exam", scope=ScopeFilter.parse({"course": "nope"})) == []


def test_doc_ids_argument_still_scopes(embedder):
    """The predecessor's API keeps working."""
    index = make_index(embedder)
    add(index, "c1", "the final exam is in week eight", doc_id="d1")
    add(index, "c2", "the final exam is in week ten", doc_id="d2")

    hits = index.search("final exam", doc_ids=["d2"])
    assert {h.chunk.doc_id for h in hits} == {"d2"}


def test_coverage_is_computed_over_the_scoped_subset(embedder):
    """The leak this project fixes.

    The predecessor computed coverage from corpus-wide document frequencies, so
    narrowing the scope to one document still inherited coverage from documents
    the filter had just excluded — a question could pass the gate on terms that
    exist only out of scope.
    """
    index = make_index(embedder)
    add(index, "c1", "the final exam is in week eight", doc_id="d1",
        metadata={"course": "cs1428"})
    add(index, "c2", "refunds for damaged footwear", doc_id="d2",
        metadata={"dept": "support"})

    scoped = index.search(
        "footwear refunds", scope=ScopeFilter.parse({"course": "cs1428"})
    )
    unscoped = index.search("footwear refunds")

    assert unscoped[0].coverage == pytest.approx(1.0)
    # Within the syllabus alone, neither term appears.
    assert not scoped or scoped[0].coverage < 1.0


def test_metadata_is_carried_onto_hits(embedder):
    index = make_index(embedder)
    add(index, "c1", "the final exam is in week eight",
        metadata={"course": "cs1428"})
    assert index.search("final exam")[0].chunk.metadata == {"course": "cs1428"}


def test_update_metadata_rescopes_existing_chunks(embedder):
    index = make_index(embedder)
    add(index, "c1", "the final exam is in week eight", doc_id="d1")
    assert index.update_metadata("d1", {"course": "cs1428"}) == 1

    hits = index.search("final exam", scope=ScopeFilter.parse({"course": "cs1428"}))
    assert len(hits) == 1


# ------------------------------------------------------------- index basics


def test_empty_index_returns_nothing(embedder):
    assert make_index(embedder).search("anything") == []


def test_stopword_only_query_returns_nothing(embedder):
    index = make_index(embedder)
    add(index, "c1", "the final exam is in week eight")
    assert index.search("the and of") == []


def test_top_k_is_respected(embedder):
    index = make_index(embedder)
    for i in range(10):
        add(index, "c{}".format(i), "exam schedule number {}".format(i),
            doc_id="d{}".format(i))
    assert len(index.search("exam schedule", top_k=3)) == 3


def test_removing_a_document_deindexes_it(embedder):
    index = make_index(embedder)
    add(index, "c1", "the final exam is in week eight", doc_id="d1")
    assert index.remove_document("d1") == 1
    assert index.is_empty
    assert index.search("final exam") == []


def test_removing_an_unknown_document_is_a_noop(embedder):
    index = make_index(embedder)
    add(index, "c1", "content", doc_id="d1")
    assert index.remove_document("nope") == 0
    assert len(index) == 1


def test_clear_empties_the_index(embedder):
    index = make_index(embedder)
    add(index, "c1", "content")
    index.clear()
    assert index.is_empty
    assert index.avg_length == 0.0


def test_doc_ids_are_listed_in_insertion_order(embedder):
    index = make_index(embedder)
    add(index, "c1", "first", doc_id="d1")
    add(index, "c2", "second", doc_id="d2")
    assert index.doc_ids() == ["d1", "d2"]
