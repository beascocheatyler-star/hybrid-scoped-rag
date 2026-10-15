"""Chunking and BM25 index behaviour."""

from __future__ import annotations

import pytest

from app.core.chunking import chunk_text
from app.core.embedding import HashingEmbedder
from app.core.retrieval import HybridIndex, Chunk, tokenize


# ------------------------------------------------------------- tokenizing

def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, WORLD!") == ["hello", "world"]


def test_tokenize_drops_stopwords_and_single_characters():
    assert tokenize("the a of x exam") == ["exam"]


def test_tokenize_keeps_stopwords_when_asked():
    assert "the" in tokenize("the exam", keep_stopwords=True)


def test_tokenize_handles_numbers_and_hyphenation():
    assert tokenize("CS-1428 scored 95%") == ["cs-1428", "scored", "95"]


def test_tokenize_empty_string():
    assert tokenize("") == []


# -------------------------------------------------------------- chunking

def test_chunk_text_returns_empty_for_blank_input():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunks_respect_the_size_limit():
    text = "\n\n".join("Paragraph number {}. {}".format(i, "word " * 60) for i in range(20))
    chunks = chunk_text(text, chunk_size=500, overlap=0)
    assert chunks
    assert all(len(c) <= 500 for c in chunks)


def test_short_text_becomes_one_chunk():
    chunks = chunk_text("A single short sentence.", chunk_size=1000, overlap=100)
    assert len(chunks) == 1


def test_overlap_carries_context_between_chunks():
    text = "\n\n".join("Section {}. {}".format(i, "filler " * 50) for i in range(6))
    without = chunk_text(text, chunk_size=400, overlap=0)
    with_overlap = chunk_text(text, chunk_size=400, overlap=100)
    assert len(with_overlap) == len(without)
    # Every chunk after the first should be longer, having gained the overlap.
    assert all(len(a) > len(b) for a, b in zip(with_overlap[1:], without[1:]))


def test_oversized_single_paragraph_is_split():
    text = "word " * 2000  # one paragraph, far over the limit
    chunks = chunk_text(text, chunk_size=300, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_a_single_enormous_word_does_not_hang():
    chunks = chunk_text("x" * 5000, chunk_size=200, overlap=0)
    assert chunks
    assert all(len(c) <= 200 for c in chunks)


def test_overlap_larger_than_chunk_size_is_clamped():
    # Must not loop forever or raise.
    chunks = chunk_text("word " * 500, chunk_size=200, overlap=500)
    assert chunks


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=0)


# ----------------------------------------------------------------- index

def make_chunk(cid: str, text: str, doc_id: str = "d1") -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=doc_id, doc_name="doc.md", position=0, text=text
    )


def test_empty_index_returns_no_hits():
    index = HybridIndex(embedder=HashingEmbedder())
    assert index.is_empty
    assert index.search("anything") == []


def test_search_ranks_the_more_relevant_chunk_first():
    index = HybridIndex(embedder=HashingEmbedder())
    index.add_chunks([
        make_chunk("a", "The final exam is cumulative and covers every chapter."),
        make_chunk("b", "Parking permits are sold at the garage office."),
    ])
    hits = index.search("final exam")
    assert hits
    assert hits[0].chunk.chunk_id == "a"


def test_search_respects_top_k():
    index = HybridIndex(embedder=HashingEmbedder())
    index.add_chunks([make_chunk(str(i), "exam policy grading") for i in range(10)])
    assert len(index.search("exam", top_k=3)) == 3


def test_search_can_be_restricted_to_documents():
    index = HybridIndex(embedder=HashingEmbedder())
    index.add_chunks([
        make_chunk("a", "exam policy", doc_id="doc1"),
        make_chunk("b", "exam policy", doc_id="doc2"),
    ])
    hits = index.search("exam", doc_ids=["doc2"])
    assert hits
    assert all(h.chunk.doc_id == "doc2" for h in hits)


def test_coverage_is_one_when_all_terms_present():
    index = HybridIndex(embedder=HashingEmbedder())
    index.add_chunks([make_chunk("a", "the final exam is cumulative")])
    hits = index.search("final exam")
    assert hits[0].coverage == pytest.approx(1.0)


def test_coverage_is_partial_when_some_terms_absent():
    index = HybridIndex(embedder=HashingEmbedder())
    index.add_chunks([make_chunk("a", "the final exam is cumulative")])
    hits = index.search("final exam parking permit")
    # "final" and "exam" present; "parking" and "permit" absent.
    assert hits[0].coverage == pytest.approx(0.5)


def test_removing_a_document_deindexes_its_chunks():
    index = HybridIndex(embedder=HashingEmbedder())
    index.add_chunks([
        make_chunk("a", "exam policy", doc_id="doc1"),
        make_chunk("b", "shipping policy", doc_id="doc2"),
    ])
    assert index.remove_document("doc1") == 1
    assert len(index) == 1
    assert not any(h.chunk.doc_id == "doc1" for h in index.search("exam"))


def test_removing_an_unknown_document_is_a_noop():
    index = HybridIndex(embedder=HashingEmbedder())
    index.add_chunks([make_chunk("a", "text")])
    assert index.remove_document("nope") == 0
    assert len(index) == 1


def test_clear_empties_the_index():
    index = HybridIndex(embedder=HashingEmbedder())
    index.add_chunks([make_chunk("a", "text here")])
    index.clear()
    assert index.is_empty
    assert index.search("text") == []


def test_document_frequencies_are_pruned_on_removal():
    """A stale term left in the frequency table would skew future IDF."""
    index = HybridIndex(embedder=HashingEmbedder())
    index.add_chunks([make_chunk("a", "zyzzyva", doc_id="doc1")])
    index.remove_document("doc1")
    assert "zyzzyva" not in index._doc_freqs


def test_stopword_only_query_returns_nothing():
    index = HybridIndex(embedder=HashingEmbedder())
    index.add_chunks([make_chunk("a", "the exam is on Monday")])
    assert index.search("the and of") == []
