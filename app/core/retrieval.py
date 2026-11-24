"""Hybrid retrieval: BM25 + dense vectors, fused, over a metadata-scoped corpus.

The predecessor ranked with BM25 alone. BM25 is a strong lexical baseline and it
fails in one specific, predictable way: it can only match words the question
actually used. Ask "when is the midterm?" of a syllabus that says "Examination
2 is held in week 8" and lexical retrieval has nothing to grab.

Dense retrieval has the mirror failure. It generalises well and matches exact
rare tokens badly — course codes, order numbers, policy window lengths, the very
strings a grounded answer must quote verbatim.

Running both and fusing the rankings covers each one's blind spot. The fusion is
**Reciprocal Rank Fusion** (Cormack et al., 2009):

    RRF(chunk) = Σ  weight_r / (K + rank_r(chunk))

over each retriever *r* the chunk appears in.

Why RRF rather than a weighted sum of the raw scores: BM25 scores are unbounded
and corpus-dependent while cosine similarities sit in [-1, 1], so combining them
numerically requires a normalisation that is itself a tuning problem, and one
that shifts every time the corpus changes. RRF reads only *ranks*, so it is
scale-free and stable — no renormalisation when documents are added. The cost is
that it discards magnitude: a runaway-best match and a merely-good one at the
same rank contribute equally. For deciding *which passages to show the model*
that is an acceptable trade.

It is not acceptable for the scope gate, which needs to know *how good* the best
match is in absolute terms, not just what ranked first. So the gate continues to
read calibrated BM25 ``normalized`` × ``coverage`` exactly as before, and
additionally considers dense similarity — see ``guard.py``. Fusion decides
ordering; calibrated scores decide whether to answer at all. Conflating those
two was the main design risk here, and keeping them separate is deliberate.
"""

from __future__ import annotations

import math
import re
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from app.core.embedding import Embedder, cosine
from app.core.metadata import ScopeFilter

# Common words carry no topical signal, so they would otherwise inflate the
# relevance of off-topic questions and defeat the scope gate.
_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because
    been before being below between both but by can cannot could couldn't did didn't
    do does doesn't doing don't down during each few for from further had hadn't has
    hasn't have haven't having he her here hers herself him himself his how i if in
    into is isn't it its itself let's me more most mustn't my myself no nor not of off
    on once only or other ought our ours ourselves out over own same shan't she should
    shouldn't so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was wasn't we were weren't what
    when where which while who whom why with won't would wouldn't you your yours
    yourself yourselves
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*")

# Standard BM25 parameters.
_K1 = 1.5
_B = 0.75

# RRF smoothing constant. 60 is the value from the original paper and the de
# facto default across implementations. Its effect: it flattens the difference
# between adjacent top ranks (1/61 vs 1/62 is a 1.6% gap) so a chunk must do
# well in *both* retrievers to beat one that is first in a single retriever.
# Lowering it sharpens the advantage of rank 1; raising it pushes toward a
# simple vote. Exposed as a setting rather than hardcoded so the behaviour is
# measurable — see scripts/measure_hybrid.py.
RRF_K_DEFAULT = 60

# NumPy is optional: with it, dense scoring is one matrix-vector product per
# query; without it, the pure-Python loop below. Small candidate sets stay on
# the loop, where NumPy's per-call overhead is not worth paying.
try:  # pragma: no cover - depends on install
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None
_NUMPY_MIN = 2000


STOPWORDS = _STOPWORDS

# Unicode tokenizer (M3). For ASCII input this is exactly the old
# ``[a-z0-9][a-z0-9'_-]*`` behaviour: a token starts with a letter or digit
# and may continue with letters, digits, apostrophes, underscores and hyphens.
# Beyond ASCII, \w covers every script, so Cyrillic, Greek, Arabic, Devanagari
# and accented Latin tokenize as words instead of vanishing (the old tokenizer
# turned "¿Cuándo?" into "cu" + "ndo" and Chinese into nothing at all).
#
# Scripts written without spaces (Han, Hiragana, Katakana, Hangul runs) would
# otherwise become one giant "word" per sentence. They are split into
# overlapping character bigrams, the standard dictionary-free approach for
# CJK retrieval; a lone character is kept as a unigram.
_UTOKEN_RE = re.compile(r"[^\W_][\w'-]*", re.UNICODE)
_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]+")


def _expand(token: str) -> List[str]:
    if not _CJK_RUN.search(token):
        return [token]
    out: List[str] = []
    pos = 0
    for m in _CJK_RUN.finditer(token):
        if m.start() > pos:
            out.append(token[pos:m.start()])
        run = m.group(0)
        out.extend([run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)])
        pos = m.end()
    if pos < len(token):
        out.append(token[pos:])
    return [t for t in out if t.strip("'-_")]


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    """NFKC-normalise, lowercase, split into word tokens (CJK as bigrams),
    drop stopwords and 1-character noise (CJK single characters are kept)."""
    norm = unicodedata.normalize("NFKC", text).lower()
    if norm.isascii():
        tokens = _TOKEN_RE.findall(norm)  # fast path, identical to before
    else:
        tokens = [t for raw in _UTOKEN_RE.findall(norm) for t in _expand(raw)]
    if keep_stopwords:
        return tokens
    return [t for t in tokens
            if t not in _STOPWORDS and (len(t) > 1 or _CJK_RUN.match(t))]


@dataclass
class Chunk:
    """One retrievable passage plus the provenance needed to cite it."""

    chunk_id: str
    doc_id: str
    doc_name: str
    position: int
    text: str
    # Document-level metadata, copied onto each chunk so that filtering is a
    # local test during the scoring loop rather than a lookup through a shared
    # dict that would have to be locked alongside the index.
    metadata: Dict[str, str] = field(default_factory=dict)
    tokens: List[str] = field(default_factory=list, repr=False)
    term_freqs: Counter = field(default_factory=Counter, repr=False)
    embedding: Optional[List[float]] = field(default=None, repr=False)
    # Anchors into the source's stored extracted text (M2). None for chunks
    # from the legacy JSON store, which never recorded positions.
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    page: Optional[int] = None
    slide: Optional[int] = None
    t_start_ms: Optional[int] = None
    t_end_ms: Optional[int] = None
    section_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = tokenize(self.text)
        if not self.term_freqs:
            self.term_freqs = Counter(self.tokens)

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class SearchHit:
    chunk: Chunk
    # The fused RRF score that determined this hit's rank. Not comparable
    # across queries and deliberately not used by the scope gate.
    score: float
    # Calibrated BM25 score in roughly 0..1, compared against a fixed relevance
    # floor regardless of query length. This is what the gate reads.
    normalized: float
    # Fraction of the query's content words that appear anywhere in the
    # (scoped) corpus. BM25 alone rewards a single rare-ish term heavily, which
    # lets "Can I return these boots?" score well against a syllabus purely on
    # the word "return". Coverage catches that.
    coverage: float = 0.0
    # Cosine similarity from the dense retriever, in roughly 0..1.
    dense_score: float = 0.0
    # Which retrievers found this chunk, and where it placed in each. Surfaced
    # through the API so a reviewer can see the fusion working rather than
    # taking it on faith.
    lexical_rank: Optional[int] = None
    dense_rank: Optional[int] = None

    @property
    def retrievers(self) -> List[str]:
        found = []
        if self.lexical_rank is not None:
            found.append("lexical")
        if self.dense_rank is not None:
            found.append("dense")
        return found


class HybridIndex:
    """In-memory hybrid index: BM25 + dense vectors, with metadata scoping.

    Thread-safe: FastAPI serves requests from a worker pool, and uploads mutate
    the index while chats read from it.
    """

    def __init__(
        self,
        embedder: Embedder,
        rrf_k: int = RRF_K_DEFAULT,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
    ):
        self._lock = threading.RLock()
        self._chunks: List[Chunk] = []
        self._matrix = None
        self._row_of: Dict[int, int] = {}
        # Inverted index: term -> chunks containing it. BM25 and coverage then
        # touch only chunks that share a term with the query, instead of
        # scanning the whole notebook (identical results, sublinear cost).
        self._postings: Dict[str, List[Chunk]] = {}
        # Insertion sequence per chunk, so postings-based scoring visits chunks
        # in the same order as the old full scan (identical tie-breaking).
        self._seq: Dict[int, int] = {}
        self._next_seq = 0
        self._cand_ids = None
        self._doc_freqs: Counter = Counter()
        self._total_length = 0
        self.embedder = embedder
        self.rrf_k = max(1, int(rrf_k))
        self.lexical_weight = max(0.0, float(lexical_weight))
        self.dense_weight = max(0.0, float(dense_weight))

    # --- introspection -------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return not self._chunks

    @property
    def avg_length(self) -> float:
        with self._lock:
            if not self._chunks:
                return 0.0
            return self._total_length / len(self._chunks)

    def doc_ids(self) -> List[str]:
        with self._lock:
            seen: Dict[str, None] = {}
            for chunk in self._chunks:
                seen.setdefault(chunk.doc_id, None)
            return list(seen)

    # --- mutation ------------------------------------------------------

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Index ``chunks``, embedding any that arrive without a vector."""
        chunks = list(chunks)
        if not chunks:
            return 0

        # Embed outside the lock: with the sentence-transformers backend this
        # is by far the slowest step, and holding the index lock through it
        # would stall every concurrent chat.
        missing = [c for c in chunks if c.embedding is None]
        if missing:
            vectors = self.embedder.embed_batch([c.text for c in missing])
            for chunk, vector in zip(missing, vectors):
                chunk.embedding = vector

        with self._lock:
            self._matrix = None
            for chunk in chunks:
                for term in chunk.term_freqs:
                    self._postings.setdefault(term, []).append(chunk)
                self._seq[id(chunk)] = self._next_seq
                self._next_seq += 1
                self._chunks.append(chunk)
                self._total_length += chunk.length
                for term in chunk.term_freqs:
                    self._doc_freqs[term] += 1
        return len(chunks)

    def remove_document(self, doc_id: str) -> int:
        """Drop every chunk belonging to ``doc_id``. Returns the number removed."""
        with self._lock:
            self._matrix = None
            keep: List[Chunk] = []
            removed = 0
            for chunk in self._chunks:
                if chunk.doc_id == doc_id:
                    removed += 1
                    self._seq.pop(id(chunk), None)
                    self._total_length -= chunk.length
                    for term in chunk.term_freqs:
                        self._doc_freqs[term] -= 1
                        if self._doc_freqs[term] <= 0:
                            del self._doc_freqs[term]
                else:
                    keep.append(chunk)
            self._chunks = keep
            if removed:
                gone = doc_id
                for term in list(self._postings):
                    lst = [c for c in self._postings[term] if c.doc_id != gone]
                    if lst:
                        self._postings[term] = lst
                    else:
                        del self._postings[term]
            return removed

    def update_metadata(self, doc_id: str, metadata: Dict[str, str]) -> int:
        """Re-tag every chunk of a document. Returns the number updated."""
        with self._lock:
            updated = 0
            for chunk in self._chunks:
                if chunk.doc_id == doc_id:
                    chunk.metadata = dict(metadata)
                    updated += 1
            return updated

    def clear(self) -> None:
        with self._lock:
            self._matrix = None
            self._postings = {}
            self._seq = {}
            self._chunks = []
            self._doc_freqs = Counter()
            self._total_length = 0

    # --- scoring internals ---------------------------------------------

    def _idf(self, term: str, n_docs: int) -> float:
        df = self._doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        # Standard BM25 IDF with +1 smoothing to keep it strictly positive.
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def _coverage(self, terms: set, candidates: Sequence[Chunk]) -> float:
        """Fraction of the query's content words present in the scoped corpus.

        Deliberately presence-based, ignoring how often a term occurs. An
        earlier version of the predecessor down-weighted terms appearing in
        many chunks, on the theory that ubiquitous terms are uninformative.
        That is true for a heterogeneous corpus and badly wrong for this one:
        in a single-subject knowledge base the most widespread term *is* the
        subject. On a 7-chunk syllabus it scored "exam" at 0.29 and "boots" at
        1.00 — precisely inverted — so questions about exams were refused while
        questions about footwear passed. Plain presence is the robust signal.

        Changed from the predecessor: coverage is computed over the *filtered*
        candidate set, not the whole corpus. The predecessor read global
        ``_doc_freqs``, so narrowing to one document still inherited
        corpus-wide coverage and a question could pass the gate on terms that
        appear only in documents the filter had just excluded — exactly the
        leak a scoped system must not have. The cost is a pass over the
        candidates to build local document frequencies, which is the same order
        as the scoring loop that follows.
        """
        if not terms:
            return 0.0
        cand_ids = self._cand_ids
        present = 0
        for term in terms:
            posting = self._postings.get(term, ())
            if cand_ids is None:
                present += 1 if posting else 0
            elif any(id(c) in cand_ids for c in posting):
                present += 1
        return present / len(terms)

    def _lexical_ranking(
        self, query_terms: List[str], candidates: Sequence[Chunk], n_docs: int
    ) -> Tuple[List[Tuple[float, Chunk]], float]:
        """BM25 over the candidate set. Returns (sorted hits, max possible score)."""
        avg_len = self._total_length / n_docs if n_docs else 0.0
        idfs = {t: self._idf(t, n_docs) for t in set(query_terms)}

        # The maximum score this query could achieve against a perfectly
        # matching chunk, used to calibrate into a comparable 0..1 range.
        max_possible = sum(idfs.values()) * (_K1 + 1)

        # Only chunks that contain at least one query term can score > 0.
        cand_ids = self._cand_ids
        seen = set()
        touched: List[Chunk] = []
        for term in set(query_terms):
            for c in self._postings.get(term, ()):
                if id(c) not in seen and (cand_ids is None or id(c) in cand_ids):
                    seen.add(id(c))
                    touched.append(c)

        seq = self._seq
        touched.sort(key=lambda c: seq.get(id(c), 0))

        scored: List[Tuple[float, Chunk]] = []
        for chunk in touched:
            score = 0.0
            for term in query_terms:
                freq = chunk.term_freqs.get(term)
                if not freq:
                    continue
                idf = idfs[term]
                if idf <= 0:
                    continue
                denom = freq + _K1 * (
                    1 - _B + _B * (chunk.length / avg_len if avg_len else 1.0)
                )
                score += idf * (freq * (_K1 + 1)) / denom
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored, max_possible

    def _dense_ranking(
        self, query: str, candidates: Sequence[Chunk]
    ) -> List[Tuple[float, Chunk]]:
        """Cosine similarity over the candidate set, best first."""
        if self.dense_weight <= 0.0:
            return []
        query_vector = self.embedder.embed_query(query)
        if not any(query_vector):
            return []

        scored: List[Tuple[float, Chunk]] = []
        for chunk in candidates:
            if chunk.embedding is None:
                continue
            similarity = cosine(query_vector, chunk.embedding)
            # Negative similarity means actively dissimilar; such a chunk
            # should not enter the ranking at all, let alone receive RRF
            # credit for placing last.
            if similarity > 0.0:
                scored.append((similarity, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    # --- vectorised path (NumPy) -----------------------------------------
    #
    # For large candidate sets, BM25 and cosine are computed with NumPy over
    # arrays rebuilt lazily after any mutation. BM25 is accumulated term by
    # term in the same operation order as the Python loop, so lexical scores
    # are bit-identical; ties are broken by insertion order, as in the loop.
    # Vectors are float32 (as stored), so dense scores can differ from the
    # Python path by float rounding (~1e-7). tests/test_m3_retrieval.py pins
    # the two paths against each other.

    def _arrays(self):
        with self._lock:
            if self._matrix is None:
                chunks = list(self._chunks)
                dim = next((len(c.embedding) for c in chunks if c.embedding is not None), 0)
                m = _np.zeros((len(chunks), dim), dtype=_np.float32)
                for i, c in enumerate(chunks):
                    if c.embedding is not None and len(c.embedding) == dim:
                        m[i] = c.embedding
                self._matrix = {
                    "chunks": chunks,
                    "row_of": {id(c): i for i, c in enumerate(chunks)},
                    "lens": _np.fromiter((c.length for c in chunks), dtype=_np.float64,
                                         count=len(chunks)),
                    "m": m,
                    "post": {},
                }
            return self._matrix

    def _posting_arrays(self, arr, term):
        cached = arr["post"].get(term)
        if cached is None:
            lst = self._postings.get(term, ())
            row_of = arr["row_of"]
            rows = _np.fromiter((row_of[id(c)] for c in lst), dtype=_np.int64, count=len(lst))
            tfs = _np.fromiter((c.term_freqs[term] for c in lst), dtype=_np.float64, count=len(lst))
            cached = arr["post"][term] = (rows, tfs)
        return cached

    def _vector_rankings(self, query_terms, query, candidates, n_docs, depth):
        arr = self._arrays()
        chunks, n = arr["chunks"], len(arr["chunks"])
        if self._cand_ids is None:
            mask = None
        else:
            mask = _np.zeros(n, dtype=bool)
            row_of = arr["row_of"]
            mask[[row_of[id(c)] for c in candidates if id(c) in row_of]] = True

        # BM25
        avg_len = self._total_length / n_docs if n_docs else 0.0
        idfs = {t: self._idf(t, n_docs) for t in set(query_terms)}
        max_possible = sum(idfs.values()) * (_K1 + 1)
        scores = _np.zeros(n, dtype=_np.float64)
        for term in query_terms:
            idf = idfs[term]
            if idf <= 0:
                continue
            rows, tfs = self._posting_arrays(arr, term)
            if rows.size == 0:
                continue
            ratio = arr["lens"][rows] / avg_len if avg_len else 1.0
            denom = tfs + _K1 * (1 - _B + _B * ratio)
            scores[rows] += idf * (tfs * (_K1 + 1)) / denom
        ok = scores > 0
        if mask is not None:
            ok &= mask
        rows = _np.nonzero(ok)[0]
        order = rows[_np.argsort(-scores[rows], kind="stable")][:depth]
        lexical = [(float(scores[i]), chunks[i]) for i in order]

        # Dense
        dense: List[Tuple[float, Chunk]] = []
        if self.dense_weight > 0.0 and arr["m"].size:
            qv = self.embedder.embed_query(query)
            if any(qv) and len(qv) == arr["m"].shape[1]:
                sims = arr["m"] @ _np.asarray(qv, dtype=_np.float32)
                ok = sims > 0
                if mask is not None:
                    ok &= mask
                rows = _np.nonzero(ok)[0]
                if rows.size > depth:
                    part = rows[_np.argpartition(-sims[rows], depth)[:depth]]
                else:
                    part = rows
                part = part[_np.lexsort((part, -sims[part]))]
                dense = [(float(sims[i]), chunks[i]) for i in part]
        return lexical, max_possible, dense

    def term_present(self, term: str, scope: Optional[ScopeFilter] = None) -> bool:
        """Does any chunk within ``scope`` contain this (tokenized) term?
        Lexical only: dense similarity says nothing about vocabulary."""
        with self._lock:
            posting = self._postings.get(term, ())
            if scope is None or scope.is_empty:
                return bool(posting)
            return any(scope.matches(c.doc_id, c.metadata) for c in posting)

    # --- query ---------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 6,
        doc_ids: Optional[List[str]] = None,
        scope: Optional[ScopeFilter] = None,
        candidate_k: Optional[int] = None,
    ) -> List[SearchHit]:
        """Return the ``top_k`` best chunks by fused rank, best first.

        ``scope`` is applied *before* scoring, so the filter constrains what can
        be retrieved rather than trimming what already was.
        """
        query_terms = tokenize(query)
        if not query_terms:
            return []

        # How deep each retriever goes before fusion. Fusing only the final
        # top_k from each would waste the complementarity that motivates
        # hybrid search: a chunk ranked 8th lexically and 2nd densely is
        # exactly the kind of result fusion should surface, and it is invisible
        # if each list is truncated at 6 first.
        depth = candidate_k if candidate_k is not None else max(top_k * 4, 20)

        with self._lock:
            if not self._chunks:
                return []

            effective_scope = scope
            if effective_scope is None and doc_ids:
                effective_scope = ScopeFilter.parse(doc_ids=doc_ids)

            if effective_scope is None or effective_scope.is_empty:
                candidates: List[Chunk] = list(self._chunks)
                self._cand_ids = None
            else:
                candidates = [
                    c
                    for c in self._chunks
                    if effective_scope.matches(c.doc_id, c.metadata)
                ]

            if not candidates:
                return []
            if effective_scope is not None and not effective_scope.is_empty:
                self._cand_ids = {id(c) for c in candidates}

            # IDF uses the global corpus size. The alternative — recomputing
            # document frequencies over the filtered set — would make a term's
            # rarity depend on the filter, so the same question against the
            # same document would score differently depending on what else was
            # in scope. Global IDF keeps BM25 scores comparable across filters,
            # which is what the calibrated gate threshold assumes.
            n_docs = len(self._chunks)

            if _np is not None and len(candidates) >= _NUMPY_MIN:
                lexical, max_possible, dense = self._vector_rankings(
                    query_terms, query, candidates, n_docs, depth)
            else:
                lexical, max_possible = self._lexical_ranking(
                    query_terms, candidates, n_docs
                )
                dense = self._dense_ranking(query, candidates)

            fused = self._fuse(lexical[:depth], dense[:depth])
            top = fused[:top_k]

            coverage = self._coverage(set(query_terms), candidates)

            hits: List[SearchHit] = []
            for entry in top:
                hits.append(
                    SearchHit(
                        chunk=entry.chunk,
                        score=entry.rrf,
                        normalized=(
                            (entry.lexical_score / max_possible)
                            if max_possible > 0
                            else 0.0
                        ),
                        coverage=coverage,
                        dense_score=entry.dense_score,
                        lexical_rank=entry.lexical_rank,
                        dense_rank=entry.dense_rank,
                    )
                )
            return hits

    def _fuse(
        self,
        lexical: Sequence[Tuple[float, Chunk]],
        dense: Sequence[Tuple[float, Chunk]],
    ) -> List["_FusedEntry"]:
        """Reciprocal Rank Fusion over the two ranked lists."""
        entries: Dict[str, _FusedEntry] = {}

        for rank, (score, chunk) in enumerate(lexical, start=1):
            entry = entries.setdefault(chunk.chunk_id, _FusedEntry(chunk=chunk))
            entry.lexical_rank = rank
            entry.lexical_score = score
            entry.rrf += self.lexical_weight / (self.rrf_k + rank)

        for rank, (score, chunk) in enumerate(dense, start=1):
            entry = entries.setdefault(chunk.chunk_id, _FusedEntry(chunk=chunk))
            entry.dense_rank = rank
            entry.dense_score = score
            entry.rrf += self.dense_weight / (self.rrf_k + rank)

        ordered = list(entries.values())
        # Ties in RRF are common (two chunks found by one retriever each, at
        # the same rank). Breaking by lexical score keeps ordering
        # deterministic, and prefers the retriever whose score the gate
        # actually reads.
        ordered.sort(key=lambda e: (e.rrf, e.lexical_score, e.dense_score), reverse=True)
        return ordered


@dataclass
class _FusedEntry:
    chunk: Chunk
    rrf: float = 0.0
    lexical_score: float = 0.0
    dense_score: float = 0.0
    lexical_rank: Optional[int] = None
    dense_rank: Optional[int] = None


# Backwards-compatible alias. The predecessor's class name appears in its tests
# and docs; keeping it importable means a reader moving between the two repos
# does not hit a spurious ImportError.
BM25Index = HybridIndex
