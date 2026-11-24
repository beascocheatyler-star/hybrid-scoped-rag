"""Query understanding: follow-ups, condensation, multi-query, reranking,
long-context selection.

The gate decision stays deterministic (invariant I5):

1. The raw question is retrieved and classified first. Greetings, meta
   questions, an empty corpus and an empty scope are decided right there.
2. A question is treated as a follow-up only if a deterministic detector
   says so: it is short, it refers back ("it", "that", "they", "what about",
   "and ...") and the conversation has an earlier question that itself passes
   the gate (recomputed, not trusted from the client).
3. If the follow-up brings its own content words, at least half of them must
   exist in the scoped notebook. "What about football?" after an in-scope
   question has a content word the notebook lacks, so it is NOT condensed and
   is gated on its own words, and refused.
4. The standalone query is built by a heuristic (previous question plus the
   follow-up) or, when QUERY_REWRITE=llm, by MODEL_REWRITE. The LLM is called
   only in this follow-up-of-an-in-scope-conversation case.
5. The standalone query is retrieved and classified deterministically. That
   result is the gate decision.

Context selection (multi-query, reranker, long-context) runs only after the
gate has allowed the question, and never feeds back into the decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from app.core.guard import ScopeDecision, ScopeResult, classify
from app.core.llm import UNTRUSTED_DATA_RULE
from app.core.metadata import ScopeFilter
from app.core.retrieval import Chunk, SearchHit, tokenize

logger = logging.getLogger(__name__)

# Referring words. Most are already stopwords (so they never count as content);
# the rest are listed so they are not mistaken for content either.
DEICTIC = frozenset("""it its it's that this these those they them their theirs there then
he him his she her one ones same former latter above previous earlier also too else""".split())
# Common non-topical English words (verbs, adverbs, adjectives, quantities,
# time units). A follow-up may add these without bringing in a new subject:
# "how quickly do I have to tell them?", "what if it's been two hours?".
# Topical nouns ("football", "taxes", "recipes") are deliberately absent, so a
# follow-up that changes the subject is still gated on its own words.
GENERIC = frozenset("""happen happens happened happening occur occurs mean means meant
skip skipped miss missed tell told say said ask asked know knew need needs needed want
wants get gets got getting give gives given take takes took taking make makes made
making do done doing use used using work works worked apply applies applied count
counts counted cost costs allowed allow allows possible okay ok fine true false right
wrong still again already yet ever never always usually often sometimes soon later
earlier late early quickly quick slowly slow fast faster long longer short shorter
much many more less fewer most least enough exactly roughly about around nearly
almost instead else otherwise anyway though really actually basically difference
different same similar happen serve serves served last lasts lasted take start starts
started end ends ended finish finished change changes changed come comes came go goes
went going keep keeps kept leave leaves left stay stays stayed find found show shows
explain describe example examples detail details reason reasons why how way ways
time times hour hours day days week weeks month months year years minute minutes
second seconds one two three four five six seven eight nine ten first second third
next previous new old older newer big small bigger smaller good bad better worse best
worst important main whole part parts kind kinds sort type types case cases thing things
""".split())

_STARTERS = ("and ", "what about", "how about", "but ", "so ", "then ", "also ", "or ",
             "why ", "how come", "and?", "more", "tell me more", "explain")
MAX_FOLLOWUP_WORDS = 14
MIN_OWN_COVERAGE = 0.5


@dataclass
class QueryPlan:
    question: str
    retrieval_query: str
    hits: List[SearchHit]
    result: ScopeResult
    matched: int
    followup: bool = False
    rewrite: str = "none"          # none | heuristic | llm
    previous_question: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def is_followup_shape(question: str) -> bool:
    q = " ".join(question.lower().split())
    words = q.replace("?", " ").replace(",", " ").split()
    if not words or len(words) > MAX_FOLLOWUP_WORDS:
        return False
    if any(w.strip(".!?'\"") in DEICTIC for w in words):
        return True
    return q.startswith(_STARTERS)


def _user_turns(history: List[Dict[str, str]]) -> List[str]:
    return [t["content"] for t in history
            if t.get("role") == "user" and (t.get("content") or "").strip()]


def plan_query(
    question: str,
    history: List[Dict[str, str]],
    store,
    relevance_floor: float,
    dense_rescue_floor: float,
    scope: Optional[ScopeFilter] = None,
    top_k: int = 6,
    rewrite_mode: str = "heuristic",
    rewriter: Optional[Callable[[str, str], str]] = None,
) -> QueryPlan:
    matched = store.count_matching(scope)

    def gate(text: str):
        hits = store.index.search(text, top_k=top_k, scope=scope)
        res = classify(text, hits, relevance_floor=relevance_floor,
                       corpus_empty=store.is_empty, scope=scope,
                       scope_matched_documents=matched,
                       dense_rescue_floor=dense_rescue_floor)
        return hits, res

    raw_hits, raw = gate(question)
    plan = QueryPlan(question, question, raw_hits, raw, matched)
    if rewrite_mode == "off" or raw.decision in (
            ScopeDecision.NO_CORPUS, ScopeDecision.EMPTY_SCOPE,
            ScopeDecision.SMALL_TALK, ScopeDecision.META):
        return plan
    if raw.decision == ScopeDecision.IN_SCOPE or not is_followup_shape(question):
        return plan

    # Find the most recent earlier user question that passes the gate itself.
    turns = _user_turns(history)
    if turns and turns[-1].strip() == question.strip():
        turns = turns[:-1]
    previous = None
    for t in reversed(turns[-4:]):
        _, r = gate(t)
        if r.decision in (ScopeDecision.IN_SCOPE, ScopeDecision.BORDERLINE):
            previous = t
            break
    if previous is None:
        plan.notes.append("follow-up shape, but no earlier in-scope question")
        return plan

    own = [t for t in tokenize(question) if t not in DEICTIC]
    topical = [t for t in own if t not in GENERIC]
    if topical:
        # Share of the follow-up's topical words that exist in the scoped
        # notebook (presence over the filtered candidates, like coverage).
        present = sum(1 for t in topical if store.index.term_present(t, scope))
        own_cov = present / len(topical)
        if own_cov < MIN_OWN_COVERAGE:
            plan.notes.append("follow-up introduces terms the notebook lacks; gated on its own words")
            return plan

    # Heuristic standalone query: the previous question plus only the
    # follow-up's topical words. Generic words ("quickly", "tell", "longer")
    # would dilute coverage without adding retrievable meaning. The model
    # still sees the user's exact wording; this string is for retrieval only.
    standalone = " ".join([previous] + topical)
    how = "heuristic"
    if rewrite_mode == "llm" and rewriter is not None:
        try:
            out = " ".join((rewriter(previous, question) or "").split())
            if out:
                standalone, how = out[:500], "llm"
        except Exception as exc:  # rewrite is best-effort; fall back
            logger.info("Query rewrite failed (%s); using heuristic", exc)

    hits, res = gate(standalone)
    if res.allows_llm and (not raw.allows_llm or res.best_score >= raw.best_score):
        return QueryPlan(question, standalone, hits, res, matched, True, how, previous,
                         plan.notes + ["condensed with the previous question"])
    return plan


REWRITE_SYSTEM = (
    "Rewrite the user's follow-up question as one standalone search query, "
    "using the previous question only to resolve references like 'it' or 'that'. "
    "Do not answer it and do not add facts. Reply with the query only. "
    + UNTRUSTED_DATA_RULE
)


def llm_rewriter(llm) -> Callable[[str, str], str]:
    def rewrite(previous: str, followup: str) -> str:
        res = llm.complete(REWRITE_SYSTEM, [{"role": "user", "content":
                           "Previous question: {}\nFollow-up: {}".format(previous, followup)}],
                           max_tokens=100)
        return res.text.strip().strip('"')
    return rewrite


# --------------------------------------------------------------------------
# Context selection (after the gate)
# --------------------------------------------------------------------------


def fuse_hit_lists(lists: List[List[SearchHit]], top_k: int, k: int = 60) -> List[SearchHit]:
    """Reciprocal Rank Fusion over several hit lists, deduplicated by chunk."""
    score: Dict[str, float] = {}
    best: Dict[str, SearchHit] = {}
    for hits in lists:
        for rank, h in enumerate(hits, start=1):
            cid = h.chunk.chunk_id
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in best:
                best[cid] = h
    order = sorted(score, key=lambda c: -score[c])
    return [best[c] for c in order[:top_k]]


MULTI_QUERY_SYSTEM = (
    "Break the user's question into at most 3 short search queries that together "
    "cover what it asks. One per line, no numbering, no answers. " + UNTRUSTED_DATA_RULE
)


def multi_query(llm, question: str) -> List[str]:
    res = llm.complete(MULTI_QUERY_SYSTEM, [{"role": "user", "content": question}], max_tokens=150)
    out = [" ".join(line.strip(" -*0123456789.").split()) for line in res.text.splitlines()]
    return [q for q in out if q][:3]


class CrossEncoderReranker:  # pragma: no cover - optional dependency
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("RERANKER=cross-encoder requires sentence-transformers") from exc
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: List[SearchHit]) -> List[SearchHit]:
        if not hits:
            return hits
        scores = self.model.predict([(query, h.chunk.text) for h in hits])
        return [h for _, h in sorted(zip(scores, hits), key=lambda x: -float(x[0]))]


_reranker = None


def get_reranker(settings):
    global _reranker
    if settings.reranker != "cross-encoder":
        return None
    if _reranker is None:
        try:
            _reranker = CrossEncoderReranker(settings.reranker_model)
        except Exception:
            logger.warning("Reranker unavailable; continuing without it.", exc_info=True)
            settings.reranker = "off"
            return None
    return _reranker


def long_context_hits(store, scope: Optional[ScopeFilter], budget_tokens: int) -> Optional[List[SearchHit]]:
    """Every in-scope chunk, in document order, if they fit the budget."""
    if budget_tokens <= 0:
        return None
    chunks: List[Chunk] = [c for c in store.index._chunks
                           if scope is None or scope.is_empty or scope.matches(c.doc_id, c.metadata)]
    if not chunks or sum(len(c.text) for c in chunks) / 4 > budget_tokens:
        return None
    docs = {d.doc_id: i for i, d in enumerate(store.list_documents())}
    chunks.sort(key=lambda c: (docs.get(c.doc_id, 0), c.position))
    return [SearchHit(chunk=c, score=0.0, normalized=0.0, coverage=0.0) for c in chunks]


def select_context(plan: QueryPlan, store, settings, scope: Optional[ScopeFilter],
                   llm=None) -> List[SearchHit]:
    """Passages to show the model. Never alters plan.result (the gate)."""
    lc = long_context_hits(store, scope, settings.long_context_token_budget)
    if lc is not None:
        plan.notes.append("long-context mode: {} passages".format(len(lc)))
        return lc
    top_k = settings.top_k
    hits = plan.hits
    if settings.multi_query and llm is not None:
        try:
            subs = multi_query(llm, plan.retrieval_query)
            lists = [hits] + [store.index.search(q, top_k=top_k, scope=scope) for q in subs]
            hits = fuse_hit_lists(lists, top_k)
            plan.notes.append("multi-query: {}".format(subs))
        except Exception as exc:
            logger.info("multi-query failed: %s", exc)
    rr = get_reranker(settings)
    if rr is not None:
        pool = store.index.search(plan.retrieval_query, top_k=30, scope=scope)
        hits = rr.rerank(plan.retrieval_query, pool)[:top_k]
        plan.notes.append("reranked top 30")
    return hits
