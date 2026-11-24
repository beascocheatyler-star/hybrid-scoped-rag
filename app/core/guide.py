"""Source Guides and automatic topic labels.

A Source Guide is a 150-250 word summary with passage citations, 5-10 key
topics and 3-5 suggested questions, stored in ``sources.guide_json``.

With an LLM configured, guides are generated map-reduce: windows of the
source's chunks are summarised (map), then the partial summaries are combined
into the guide (reduce). A source that fits one window skips the map step.
Without an LLM, a deterministic extractive guide is built instead (salient
sentences in document order, frequent terms as topics, templated questions),
so the feature works offline at zero cost.

Auto-labelling: once a notebook has AUTO_LABEL_MIN (5) ready sources with
guides, sources are grouped into topic labels written to metadata key
``label``. That reuses the existing facet and scope machinery, so labels
become filter chips for free. A label the user edited is never overwritten:
the guide records the label it last applied, and only a label still equal to
that value (or absent) is replaced.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Dict, List, Optional

from app.core.llm import UNTRUSTED_DATA_RULE, LLMRefusalError, LLMUnavailableError
from app.core.retrieval import STOPWORDS, tokenize

AUTO_LABEL_MIN = 5
WINDOW_CHARS = 24000
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_MARK = re.compile(r"\[(\d{1,3})\]")

GUIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "topics", "questions"],
    "additionalProperties": False,
}

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label", "source_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# Extractive fallback
# --------------------------------------------------------------------------


_WEAK = frozenset("""will would could should shall must might also upon even much many
one two three first second may made make makes like well every great little thus hence
however therefore though although whether within without another other others some""".split())


def _top_terms(text: str, k: int) -> List[str]:
    words = [w for w in tokenize(text) if len(w) > 3 and not w.isdigit() and w not in _WEAK]
    counts = Counter(words)
    # Prefer repeated two-word phrases, which read better as topics.
    toks = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]+", text)]
    low = [t.lower() for t in toks]
    bigrams = Counter(
        (a, b) for a, b in zip(low, low[1:])
        if a not in STOPWORDS and b not in STOPWORDS and a not in _WEAK and b not in _WEAK
        and len(a) > 2 and len(b) > 2
    )
    topics: List[str] = []
    for (a, b), n in bigrams.most_common(k * 3):
        if n < 2:
            break
        topics.append("{} {}".format(a, b))
        if len(topics) >= k // 2:
            break
    used = {w for t in topics for w in t.split()}
    for w, _ in counts.most_common(k * 3):
        if w not in used:
            topics.append(w)
            used.add(w)
        if len(topics) >= k:
            break
    return topics[:k]


def extractive_guide(chunks: List[Dict]) -> Dict:
    """chunks: [{"id", "text"}] in document order."""
    text = "\n\n".join(c["text"] for c in chunks)
    freq = Counter(w for w in tokenize(text) if len(w) > 2)
    sents = []
    for ci, c in enumerate(chunks):
        prose = " ".join(
            ln for ln in c["text"].splitlines()
            if ln.strip() and not re.match(r"\s*(#|\||[-*]\s|\d+\.\s|---)", ln))
        for s in _SENT.split(" ".join(prose.split())):
            s = s.strip()
            words = tokenize(s)
            if len(s) < 40 or len(s) > 400 or len(words) < 5 or s.startswith("#"):
                continue
            score = sum(math.log1p(freq[w]) for w in set(words)) / (1 + math.log1p(len(words)))
            sents.append((score, len(sents), ci, s))
    chosen, seen, total = [], set(), 0
    for score, order, ci, s in sorted(sents, key=lambda x: -x[0]):
        if s in seen:
            continue
        chosen.append((order, ci, s))
        seen.add(s)
        total += len(s.split())
        if total >= 180:
            break
    chosen = [(ci, s) for _, ci, s in sorted(chosen)]  # back to document order
    # Cite each sentence with the 1-based number of its chunk.
    summary = " ".join("{} [{}]".format(s, ci + 1) for ci, s in chosen)
    topics = _top_terms(text, 8)
    questions = ["What does this source say about {}?".format(t) for t in topics[:4]]
    return {"summary": summary, "topics": topics, "questions": questions,
            "generated_by": "extractive"}


# --------------------------------------------------------------------------
# LLM map-reduce
# --------------------------------------------------------------------------

_GUIDE_SYSTEM = (
    "You write source guides for a research notebook. You are given numbered "
    "SOURCE passages from ONE document. Produce: a 150-250 word summary in which "
    "every sentence that states a fact from the source ends with the bracketed "
    "number of the passage that supports it, e.g. 'The final is in week 16 [2].'; "
    "5-10 short key topics; and 3-5 questions a reader could ask that this source "
    "answers. Use only the passages. " + UNTRUSTED_DATA_RULE
)

_MAP_SYSTEM = (
    "Summarise this portion of a document in at most 150 words, keeping concrete "
    "facts, names, numbers and dates. " + UNTRUSTED_DATA_RULE
)


def _numbered(chunks: List[Dict], offset: int = 0) -> str:
    return "\n\n".join("SOURCE [{}]\n{}".format(i + 1 + offset, c["text"]) for i, c in enumerate(chunks))


def llm_guide(chunks: List[Dict], title: str, llm) -> Dict:
    total = sum(len(c["text"]) for c in chunks)
    if total <= WINDOW_CHARS:
        body = _numbered(chunks)
    else:
        windows, cur, size = [], [], 0
        for c in chunks:
            if cur and size + len(c["text"]) > WINDOW_CHARS:
                windows.append(cur)
                cur, size = [], 0
            cur.append(c)
            size += len(c["text"])
        if cur:
            windows.append(cur)
        partials = []
        for w in windows:
            res = llm.complete(_MAP_SYSTEM, [{"role": "user", "content": _numbered(w)}], max_tokens=600)
            partials.append(res.text)
        # Reduce over partial summaries plus the first passage of each window,
        # so the summary can still cite real passages.
        heads = [w[0] for w in windows]
        body = ("PARTIAL SUMMARIES (for coverage, not citable):\n" + "\n\n".join(partials)
                + "\n\nCITABLE PASSAGES:\n" + _numbered(heads))
        chunks = heads
    data, _ = llm.complete_json(
        _GUIDE_SYSTEM,
        [{"role": "user", "content": "Document title: {}\n\n{}".format(title, body)}],
        GUIDE_SCHEMA, max_tokens=2000)
    return {
        "summary": str(data.get("summary", "")).strip(),
        "topics": [str(t).strip() for t in data.get("topics", []) if str(t).strip()][:10],
        "questions": [str(q).strip() for q in data.get("questions", []) if str(q).strip()][:5],
        "generated_by": "llm",
        "_cited_chunks": chunks,
    }


def build_guide(chunks: List[Dict], title: str, llm=None) -> Dict:
    """Return the guide dict; citations map [n] markers to chunk ids."""
    cited = chunks
    guide = None
    if llm is not None and chunks:
        try:
            guide = llm_guide(chunks, title, llm)
            cited = guide.pop("_cited_chunks")
        except (LLMUnavailableError, LLMRefusalError, ValueError, KeyError):
            guide = None
    if guide is None:
        guide = extractive_guide(chunks)
    cites = []
    for m in _MARK.finditer(guide["summary"]):
        i = int(m.group(1)) - 1
        if 0 <= i < len(cited):
            cites.append({"marker": int(m.group(1)), "chunk_id": cited[i]["id"]})
    # Drop markers that point at nothing (hallucinated citations).
    valid = {c["marker"] for c in cites}
    guide["summary"] = _MARK.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "",
                                 guide["summary"]).replace(" .", ".").strip()
    guide["citations"] = cites
    return guide


# --------------------------------------------------------------------------
# Auto-labels
# --------------------------------------------------------------------------


def _fold(word: str) -> str:
    w = word.lower()
    for suf in ("oes", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)] if suf != "oes" else w[:-2]
    return w


def extractive_labels(sources: List[Dict]) -> Dict[str, str]:
    """sources: [{"id", "title", "topics"}] -> {source_id: label}.

    Each source's topics are reduced to single, plural-folded terms; a source
    is labelled with its term shared by the most sources (ties: earlier
    topic). Terms unique to one source fall back to its first topic."""
    terms = {}
    for s in sources:
        seen: List[str] = []
        for t in s["topics"][:8]:
            for w in t.split():
                f = _fold(w)
                if len(f) > 2 and f not in seen:
                    seen.append(f)
        terms[s["id"]] = seen
    pool = Counter(t for ts in terms.values() for t in ts)
    # Global prominence (earlier topics weigh more) breaks ties the same way
    # for every source, so related sources converge on one label.
    weight = Counter()
    for ts in terms.values():
        for i, t in enumerate(ts):
            weight[t] += len(ts) - i
    out = {}
    for s in sources:
        ts = terms[s["id"]]
        shared = [t for t in ts if pool[t] > 1]
        pick = max(shared, key=lambda t: (pool[t], weight[t], t)) if shared else (
            ts[0] if ts else "general")
        out[s["id"]] = pick.title()
    return out


def llm_labels(sources: List[Dict], llm) -> Dict[str, str]:
    listing = "\n".join("- id={} | title={} | topics={} | summary={}".format(
        s["id"], s["title"], ", ".join(s["topics"][:8]), s["summary"][:400]) for s in sources)
    data, _ = llm.complete_json(
        "Group these notebook sources into 2-6 short topic labels (1-3 words each). "
        "Every source id must appear in exactly one group. " + UNTRUSTED_DATA_RULE,
        [{"role": "user", "content": "SOURCES:\n" + listing}], LABEL_SCHEMA, max_tokens=1000)
    ids = {s["id"] for s in sources}
    out = {}
    for g in data.get("labels", []):
        label = " ".join(str(g.get("label", "")).split())[:40]
        for sid in g.get("source_ids", []):
            if sid in ids and label:
                out[sid] = label
    return out


def compute_labels(sources: List[Dict], llm=None) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if llm is not None:
        try:
            labels = llm_labels(sources, llm)
        except (LLMUnavailableError, LLMRefusalError, ValueError, KeyError):
            labels = {}
    fallback = extractive_labels(sources)
    return {s["id"]: labels.get(s["id"]) or fallback[s["id"]] for s in sources}


def dumps(d: Dict) -> str:
    return json.dumps(d, ensure_ascii=False)
