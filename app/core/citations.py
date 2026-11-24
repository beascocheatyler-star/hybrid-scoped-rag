"""Verifiable passage citations (roadmap C1-C4).

C1  The context block numbers passages; the prompt asks for a bracketed
    marker after every sentence stating a fact from them: "... week 16 [2]."
    Several markers are allowed ("[1][3]" or "[1, 3]").
C2  ``parse`` extracts markers and maps each to its passage (chunk id, source
    id, page/slide/timestamp, offsets). A marker pointing at no provided
    passage is removed from the text and logged as ``hallucinated_citation``.
C3  ``verify`` scores every (sentence, cited passage) pair: lexical overlap of
    content words plus embedding cosine. Below SUPPORT_FLOOR the citation is
    ``weak``; a sentence whose citations are all weak is ``unsupported``.
C4  Returned citations are the cited passages only, in first-marker order,
    each with a quote: the best-matching sentence inside the passage, for the
    hover card.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from app.core.embedding import Embedder, cosine
from app.core.retrieval import SearchHit, tokenize

logger = logging.getLogger(__name__)

# Calibrated by scripts/measure_support.py on the eval corpora (see
# docs/design/M4.md). Overridable with SUPPORT_FLOOR.
SUPPORT_FLOOR_DEFAULT = 0.50
LEX_WEIGHT = 0.6
COS_WEIGHT = 0.4

# [3], [1][3], [1, 3], [1,2,5] — but not markdown links "[text](url)" or
# footnote-like words.
_MARKER = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\](?!\()")


@dataclass
class Citation:
    marker: int
    chunk_id: str
    source_id: str
    source_title: str
    position: int
    quote: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    page: Optional[int] = None
    slide: Optional[int] = None
    t_start_ms: Optional[int] = None
    t_end_ms: Optional[int] = None
    section_path: Optional[str] = None
    support: float = 0.0
    weak: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Sentence:
    text: str
    markers: List[int] = field(default_factory=list)
    supported: bool = True
    support: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CitedAnswer:
    text: str                         # answer with hallucinated markers removed
    citations: List[Citation]         # cited passages only, first-marker order
    sentences: List[Sentence]
    hallucinated: List[int] = field(default_factory=list)

    @property
    def unsupported_count(self) -> int:
        return sum(1 for s in self.sentences if s.markers and not s.supported)


def _content(text: str) -> set:
    return {t for t in tokenize(text) if len(t) > 2 or not t.isascii()}


_TRAILING_MARKERS = re.compile(r"([.!?])((?:\s*\[\d{1,3}(?:\s*,\s*\d{1,3})*\](?!\())+)")
_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(*#\-])")
_LIST_START = re.compile(r"^\s*(?:[-*+]\s|#{1,6}\s|\||\d+[.)]\s)")


def split_sentences(text: str) -> List[str]:
    """Split into sentences, keeping citation markers with the sentence they
    follow ("... week 16. [2] Next" -> "... week 16 [2]." + "Next") and
    joining soft-wrapped lines (a list item, heading or table row starts a new
    segment)."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        segs: List[str] = []
        for line in block.split("\n"):
            if not line.strip():
                continue
            if segs and not _LIST_START.match(line):
                segs[-1] += " " + line.strip()
            else:
                segs.append(line.strip())
        for seg in segs:
            seg = _TRAILING_MARKERS.sub(lambda m: " " + m.group(2).strip() + m.group(1), seg)
            out.extend(x.strip() for x in _BREAK.split(seg) if x.strip())
    return out


def markers_in(text: str) -> List[int]:
    found = []
    for m in _MARKER.finditer(text):
        for part in m.group(1).split(","):
            found.append(int(part.strip()))
    return found


def best_quote(sentence: str, passage: str, limit: int = 300) -> str:
    """The sentence of ``passage`` that best matches ``sentence``."""
    target = _content(sentence)
    best, best_score = "", -1.0
    for cand in split_sentences(passage) or [passage]:
        words = _content(cand)
        score = len(target & words) / (len(words) ** 0.5 + 1) if words else 0.0
        if score > best_score:
            best, best_score = cand, score
    best = " ".join(best.split())
    return best if len(best) <= limit else best[:limit].rstrip() + "…"


def support_score(sentence: str, passage: str, embedder: Optional[Embedder] = None,
                  sentence_vec=None, passage_vec=None) -> float:
    s = _content(_MARKER.sub("", sentence))
    if not s:
        return 1.0  # nothing factual to support
    lex = len(s & _content(passage)) / len(s)
    cos = 0.0
    if embedder is not None:
        a = sentence_vec if sentence_vec is not None else embedder.embed(sentence)
        b = passage_vec if passage_vec is not None else embedder.embed(passage)
        cos = max(0.0, cosine(a, b))
    return LEX_WEIGHT * lex + COS_WEIGHT * cos


def process(answer: str, passages: Sequence[SearchHit], embedder: Optional[Embedder] = None,
            support_floor: float = SUPPORT_FLOOR_DEFAULT) -> CitedAnswer:
    """Parse, validate and verify the citations in a model answer."""
    k = len(passages)
    hallucinated: List[int] = []

    def fix(m):
        keep = []
        for part in m.group(1).split(","):
            n = int(part.strip())
            if 1 <= n <= k:
                if n not in keep:
                    keep.append(n)
            else:
                hallucinated.append(n)
        return "".join("[{}]".format(n) for n in keep)

    text = _MARKER.sub(fix, answer)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)  # tidy " ." left by removals
    if hallucinated:
        logger.warning("hallucinated_citation markers=%s passages=%d", hallucinated, k)

    sentences: List[Sentence] = []
    order: List[int] = []
    support_by_marker: Dict[int, float] = {}
    pvecs: Dict[int, List[float]] = {}
    for raw in split_sentences(text):
        ms = []
        for n in markers_in(raw):
            if n not in ms:
                ms.append(n)
        clean = re.sub(r"\s+([.,;:!?])", r"\1", _MARKER.sub("", raw)).strip()
        sent = Sentence(text=clean, markers=ms)
        if ms:
            svec = embedder.embed(sent.text) if embedder is not None else None
            best = 0.0
            for n in ms:
                if n not in order:
                    order.append(n)
                chunk = passages[n - 1].chunk
                if embedder is not None and n not in pvecs:
                    pvecs[n] = chunk.embedding or embedder.embed(chunk.text)
                sc = support_score(sent.text, chunk.text, embedder, svec, pvecs.get(n))
                support_by_marker[n] = max(support_by_marker.get(n, 0.0), sc)
                best = max(best, sc)
            sent.support = round(best, 4)
            sent.supported = best >= support_floor
        sentences.append(sent)

    citations: List[Citation] = []
    for n in order:
        ch = passages[n - 1].chunk
        cited_sents = [s.text for s in sentences if n in s.markers]
        sc = support_by_marker.get(n, 0.0)
        citations.append(Citation(
            marker=n, chunk_id=ch.chunk_id, source_id=ch.doc_id, source_title=ch.doc_name,
            position=ch.position, quote=best_quote(" ".join(cited_sents), ch.text),
            start_char=ch.start_char, end_char=ch.end_char, page=ch.page, slide=ch.slide,
            t_start_ms=ch.t_start_ms, t_end_ms=ch.t_end_ms, section_path=ch.section_path,
            support=round(sc, 4), weak=sc < support_floor,
        ))
    return CitedAnswer(text=text.strip(), citations=citations, sentences=sentences,
                       hallucinated=hallucinated)


CITATION_RULES = (
    "Cite your sources with bracketed passage numbers. The CONTEXT passages are "
    "labelled [Source 1], [Source 2], ...; after every sentence that states a fact "
    "taken from them, put the number(s) of the supporting passage(s) in square "
    "brackets, e.g. \"The final exam is in week 16 [2].\" or \"... [1][3].\" Only "
    "cite passages that were provided. Sentences that state no fact from the "
    "passages (transitions, refusals, questions back to the user) carry no marker."
)
