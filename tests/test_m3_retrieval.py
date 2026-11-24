"""M3: Unicode tokenizer, vectorised scoring, gate fix, follow-ups, context
selection, calibrations."""

from __future__ import annotations

import random

import pytest

import app.core.retrieval as R
from app.config import Settings
from app.core.embedding import HashingEmbedder
from app.core.guard import ScopeDecision, classify
from app.core.metadata import ScopeFilter
from app.core.query import fuse_hit_lists, is_followup_shape, long_context_hits, plan_query
from app.core.retrieval import Chunk, HybridIndex, SearchHit, tokenize
from app.core.services import build_services

SYL = (
    "# CS 1428 Syllabus\n\nExam 1 is on February 18 and covers chapters 1 to 5. "
    "Exam 2 covers chapters 6 to 9. The final exam is cumulative.\n\n"
    "Late work loses ten percent per day. Each student has three late days.\n"
)


# ------------------------------------------------------------- tokenizer


def test_ascii_tokenization_unchanged():
    old = R._TOKEN_RE
    for text in ["Zebra-print boots, don't worry!", "CS1428 exam_1 2026", "It's O'Brien's x-ray"]:
        expected = [t for t in old.findall(text.lower()) if t not in R._STOPWORDS and len(t) > 1]
        assert tokenize(text) == expected


def test_unicode_tokenization():
    assert tokenize("¿Cuándo es el examen final?") == ["cuándo", "es", "el", "examen", "final"]
    assert tokenize("Как вернуть товар?") == ["как", "вернуть", "товар"]
    assert tokenize("期末考试") == ["期末", "末考", "考试"]
    assert tokenize("猫") == ["猫"]
    assert tokenize("ｆｕｌｌｗｉｄｔｈ") == ["fullwidth"]  # NFKC


def test_non_latin_question_over_non_latin_corpus_is_answered(tmp_path):
    """Known defect #4: non-Latin questions were refused as 'no searchable
    terms'. Now a Chinese question over a Chinese source passes the gate."""
    idx = HybridIndex(HashingEmbedder(dimensions=256))
    idx.add_chunks([
        Chunk("c1", "d", "zh.md", 0, "期末考试在五月十一日上午八点举行。考试是累积的。"),
        Chunk("c2", "d", "zh.md", 1, "迟交作业每天扣百分之十。每个学生有三天宽限期。"),
    ])
    q = "期末考试是什么时候？"
    hits = idx.search(q)
    assert hits and hits[0].chunk.chunk_id == "c1"
    r = classify(q, hits, relevance_floor=0.42, corpus_empty=False)
    assert r.allows_llm and r.decision != ScopeDecision.OUT_OF_SCOPE
    off = classify("足球比赛谁赢了？", idx.search("足球比赛谁赢了？"), relevance_floor=0.42, corpus_empty=False)
    assert off.decision == ScopeDecision.OUT_OF_SCOPE


# ------------------------------------------------------------- vector path


def _corpus(n, seed=3):
    rng = random.Random(seed)
    vocab = ["w{}".format(i) for i in range(400)]
    weights = [1.0 / (i + 1) for i in range(400)]
    return [Chunk("c{}".format(i), "d{}".format(i % 7), "d.md", i,
                  " ".join(rng.choices(vocab, weights, k=60)),
                  metadata={"g": "a" if i % 3 else "b"}) for i in range(n)]


@pytest.mark.skipif(R._np is None, reason="numpy not installed")
@pytest.mark.parametrize("scoped", [False, True])
def test_numpy_path_matches_python_path(monkeypatch, scoped):
    emb = HashingEmbedder(dimensions=128)
    chunks = _corpus(300)
    py = HybridIndex(emb)
    py.add_chunks([Chunk(c.chunk_id, c.doc_id, c.doc_name, c.position, c.text, dict(c.metadata)) for c in chunks])
    vec = HybridIndex(emb)
    vec.add_chunks(chunks)
    scope = ScopeFilter.parse({"g": "a"}) if scoped else None
    rng = random.Random(9)
    for _ in range(25):
        q = " ".join("w{}".format(rng.randint(0, 120)) for _ in range(3))
        monkeypatch.setattr(R, "_NUMPY_MIN", 10 ** 9)
        a = py.search(q, top_k=8, scope=scope)
        monkeypatch.setattr(R, "_NUMPY_MIN", 1)
        b = vec.search(q, top_k=8, scope=scope)
        assert [h.chunk.chunk_id for h in a] == [h.chunk.chunk_id for h in b], q
        for x, y in zip(a, b):
            assert x.normalized == y.normalized          # BM25 bit-identical
            assert x.coverage == y.coverage
            assert abs(x.dense_score - y.dense_score) < 1e-5  # float32 vectors
            assert (x.lexical_rank, x.dense_rank) == (y.lexical_rank, y.dense_rank)


def test_postings_follow_removal():
    idx = HybridIndex(HashingEmbedder(dimensions=64))
    idx.add_chunks([Chunk("a", "d1", "x", 0, "alpha beta"), Chunk("b", "d2", "y", 0, "beta gamma")])
    assert idx.term_present("alpha")
    idx.remove_document("d1")
    assert not idx.term_present("alpha") and idx.term_present("beta")
    assert [h.chunk.chunk_id for h in idx.search("alpha beta")] == ["b"]
    assert idx.term_present("gamma", ScopeFilter.parse(doc_ids=["d2"]))
    assert not idx.term_present("gamma", ScopeFilter.parse(doc_ids=["d1"]))


# ------------------------------------------------------------- gate fix


def test_gate_reads_strongest_lexical_hit_not_fused_top():
    """Known defect #3: a dense-only hit fused to rank 1 (normalized 0) used to
    sink relevance even when a lexically strong hit sat at rank 2."""
    c1 = Chunk("dense", "d", "x", 0, "unrelated text")
    c2 = Chunk("lex", "d", "x", 1, "final exam week sixteen")
    hits = [SearchHit(c1, score=0.033, normalized=0.0, coverage=1.0, dense_score=0.4, dense_rank=1),
            SearchHit(c2, score=0.032, normalized=0.9, coverage=1.0, lexical_rank=1)]
    r = classify("final exam week", hits, relevance_floor=0.42, corpus_empty=False)
    assert r.decision == ScopeDecision.IN_SCOPE
    assert r.best_score == pytest.approx(0.9 ** 0.5)


# ------------------------------------------------------------- follow-ups


@pytest.fixture
def store(tmp_path):
    s = Settings()
    s.data_dir = tmp_path
    s.source_guides = False
    svc = build_services(s, HashingEmbedder(dimensions=256))
    st = svc.registry.default_store()
    st.add("syllabus.md", SYL.encode())
    yield st
    svc.close()


def _hist(*qs):
    out = []
    for q in qs:
        out += [{"role": "user", "content": q}, {"role": "assistant", "content": "ok"}]
    return out


def test_followup_shape():
    assert is_followup_shape("What chapters does it cover?")
    assert is_followup_shape("And Exam 2?")
    assert not is_followup_shape("What is the late work policy for the programming assignments in this course this term?")


def test_followup_condensed_and_answered(store):
    raw = plan_query("When is it?", [], store, 0.42, 0.36)
    assert raw.result.decision == ScopeDecision.OUT_OF_SCOPE
    p = plan_query("When is it?", _hist("When is the final exam?"), store, 0.42, 0.36)
    assert p.followup and p.rewrite == "heuristic"
    assert p.result.allows_llm
    assert "final exam" in p.retrieval_query


def test_offtopic_followup_still_refused(store):
    p = plan_query("What about football?", _hist("When is the final exam?"), store, 0.42, 0.36)
    assert not p.followup and p.result.decision == ScopeDecision.OUT_OF_SCOPE


def test_followup_after_offtopic_turn_is_not_condensed(store):
    p = plan_query("When is it?", _hist("Who won the 1998 World Cup?"), store, 0.42, 0.36)
    assert p.result.decision == ScopeDecision.OUT_OF_SCOPE


def test_llm_rewriter_used_only_for_followups(store):
    calls = []

    def rewriter(prev, q):
        calls.append((prev, q))
        return "late days per student"

    plan_query("What about football?", _hist("When is the final exam?"), store, 0.42, 0.36,
               rewrite_mode="llm", rewriter=rewriter)
    plan_query("Who won the World Cup?", [], store, 0.42, 0.36, rewrite_mode="llm", rewriter=rewriter)
    assert calls == []  # I5: off-topic and first turns never reach the LLM
    p = plan_query("How many are there?", _hist("What are late days?"), store, 0.42, 0.36,
                   rewrite_mode="llm", rewriter=rewriter)
    assert calls and p.rewrite == "llm" and p.result.allows_llm


def test_engine_followup_offtopic_zero_llm_calls(store, tmp_path):
    from app.core.engine import ChatEngine
    from app.core.persona import PRESETS
    from tests.fakes import FakeLLM

    s = Settings()
    s.anthropic_api_key = "x"
    eng = ChatEngine(s, store)
    eng._client = FakeLLM()
    a = eng.answer("What about football?", PRESETS["course_ta"], history=_hist("When is the final exam?"))
    assert a.decision == ScopeDecision.OUT_OF_SCOPE and eng._client.call_count == 0
    b = eng.answer("When is it?", PRESETS["course_ta"], history=_hist("When is the final exam?"))
    assert b.in_scope and eng._client.call_count == 1 and b.rewrite == "heuristic"


# ------------------------------------------------------------- context selection


def test_rrf_fusion_of_hit_lists():
    a, b, c = (Chunk(x, "d", "n", i, x) for i, x in enumerate("abc"))
    l1 = [SearchHit(a, 0, 0), SearchHit(b, 0, 0)]
    l2 = [SearchHit(b, 0, 0), SearchHit(c, 0, 0)]
    assert [h.chunk.chunk_id for h in fuse_hit_lists([l1, l2], 3)] == ["b", "a", "c"]


def test_long_context_mode(store):
    assert long_context_hits(store, None, 0) is None
    hits = long_context_hits(store, None, 100000)
    assert hits and [h.chunk.position for h in hits] == sorted(h.chunk.position for h in hits)
    assert long_context_hits(store, None, 5) is None  # does not fit the budget
    assert long_context_hits(store, ScopeFilter.parse({"x": "y"}), 100000) is None


def test_context_selection_never_changes_gate(store):
    from app.core.query import select_context

    s = Settings()
    s.long_context_token_budget = 100000
    p = plan_query("When is the final exam?", [], store, 0.42, 0.36)
    before = (p.result.decision, p.result.best_score)
    ctx = select_context(p, store, s, None)
    assert len(ctx) >= 1 and (p.result.decision, p.result.best_score) == before


# ------------------------------------------------------------- calibrations


def test_calibration_row_overrides_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("RELEVANCE_FLOOR", raising=False)
    monkeypatch.delenv("DENSE_RESCUE_FLOOR", raising=False)
    s = Settings()
    s.data_dir = tmp_path
    svc = build_services(s, HashingEmbedder(dimensions=64))
    svc.db.execute("INSERT INTO calibrations(embedder_signature, relevance_floor, dense_rescue_floor, "
                   "measured_at) VALUES ('hashing:64', 0.47, 0.5, 'now')")
    svc.close()
    s2 = Settings()
    s2.data_dir = tmp_path
    svc2 = build_services(s2, HashingEmbedder(dimensions=64))
    assert (s2.relevance_floor, s2.dense_rescue_floor) == (0.47, 0.5)
    svc2.close()
    monkeypatch.setenv("RELEVANCE_FLOOR", "0.44")  # explicit env wins
    s3 = Settings()
    s3.data_dir = tmp_path
    svc3 = build_services(s3, HashingEmbedder(dimensions=64))
    assert s3.relevance_floor == 0.44 and s3.dense_rescue_floor == 0.5
    svc3.close()


def test_uncalibrated_semantic_embedder_disables_rescue(tmp_path, monkeypatch):
    monkeypatch.delenv("DENSE_RESCUE_FLOOR", raising=False)

    class FakeST(HashingEmbedder):
        name = "sentence-transformers:fake"

    s = Settings()
    s.data_dir = tmp_path
    svc = build_services(s, FakeST(dimensions=64))
    assert s.dense_rescue_floor == 1.0
    svc.close()
