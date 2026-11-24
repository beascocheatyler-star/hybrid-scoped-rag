"""Studio artifacts: reports, data tables, mind maps, flashcards, quizzes.

Pipeline (one job per artifact):

1. **Gate.** A focus prompt, if given, must pass the same deterministic scope
   gate as chat. An off-topic focus is refused before any LLM call (I5).
2. **Passages.** Up to MAX_PASSAGES chunks are selected from the SELECTED
   sources only (I1), by retrieval on the focus prompt or spread evenly across
   the sources. They are numbered [1]..[k], as in chat.
3. **Map.** The per-source Source Guides (already generated, already
   map-reduced) give the reduce step a summary of every selected source.
4. **Reduce.** An artifact-specific prompt asks for JSON in a fixed schema,
   with every item citing passage numbers. Without an LLM a deterministic
   generator builds the same structure from the passages.
5. **Validate.** The JSON is parsed into pydantic models; citation numbers are
   mapped to passages; items with no valid citation are dropped (every
   remaining item cites the selected sources). Size and depth bounds are
   enforced.

The artifact stores the prompt used, the source-id snapshot and the
citations, so it never changes when sources change later; the API marks it
``stale`` when the source set moves.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from app.core.citations import best_quote, process as process_citations, split_sentences
from app.core.guard import classify
from app.core.llm import UNTRUSTED_DATA_RULE, LLMRefusalError, LLMUnavailableError
from app.core.metadata import ScopeFilter
from app.core.retrieval import STOPWORDS, Chunk, tokenize

logger = logging.getLogger(__name__)

TYPES = ("report", "data_table", "mind_map", "flashcards", "quiz")
REPORT_FORMATS = ("briefing", "study_guide", "faq", "timeline", "blog_post", "custom")
QUIZ_TYPES = ("multiple_choice", "multi_select", "short_answer", "fill_blank")
QUANTITY = {"fewer": 10, "standard": 20, "more": 40}
MAX_PASSAGES = 40
MIND_MAP_MAX_NODES, MIND_MAP_MIN_NODES, MIND_MAP_MAX_DEPTH = 80, 20, 4


class StudioRefused(Exception):
    """The focus prompt failed the scope gate (no LLM call was made)."""


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class Cite(BaseModel):
    marker: int
    chunk_id: str
    source_id: str
    source_title: str
    quote: str = ""
    page: Optional[int] = None
    slide: Optional[int] = None
    t_start_ms: Optional[int] = None
    section_path: Optional[str] = None


class ReportSection(BaseModel):
    heading: str
    body_md: str
    citations: List[Cite] = Field(default_factory=list)


class Report(BaseModel):
    title: str
    format: str
    sections: List[ReportSection]


class Cell(BaseModel):
    text: str
    citations: List[Cite] = Field(default_factory=list)


class TableRow(BaseModel):
    id: str
    cells: List[Cell]
    citations: List[Cite] = Field(default_factory=list)


class DataTable(BaseModel):
    title: str
    columns: List[str]
    rows: List[TableRow]


class MindNode(BaseModel):
    id: str
    label: str
    citations: List[Cite] = Field(default_factory=list)
    children: List["MindNode"] = Field(default_factory=list)


MindNode.model_rebuild()


class MindMap(BaseModel):
    title: str
    root: MindNode


class Flashcard(BaseModel):
    id: str
    front: str
    back: str
    citations: List[Cite]


class Flashcards(BaseModel):
    title: str
    difficulty: str
    cards: List[Flashcard]


class QuizItem(BaseModel):
    id: str
    type: str
    question: str
    options: List[str] = Field(default_factory=list)
    answer: Any  # int (MC), list[int] (multi-select), str (short / fill)
    rationale: str
    citations: List[Cite]


class Quiz(BaseModel):
    title: str
    difficulty: str
    items: List[QuizItem]


# --------------------------------------------------------------------------
# Passages
# --------------------------------------------------------------------------


def select_passages(store, source_ids: Optional[List[str]], focus: str = "",
                    k: int = MAX_PASSAGES) -> List[Chunk]:
    """Chunks from the selected sources only (I1)."""
    scope = ScopeFilter.parse(doc_ids=source_ids) if source_ids else None
    allowed = {d.doc_id for d in store.list_documents()
               if scope is None or scope.matches(d.doc_id, d.metadata)}
    pool = [c for c in store.index._chunks if c.doc_id in allowed]
    if not pool:
        return []
    if focus.strip():
        hits = store.index.search(focus, top_k=k, scope=scope)
        chosen = [h.chunk for h in hits if h.chunk.doc_id in allowed]
        if chosen:
            return chosen
    # Even spread: round-robin over sources, evenly spaced positions.
    by_src: Dict[str, List[Chunk]] = {}
    for c in pool:
        by_src.setdefault(c.doc_id, []).append(c)
    for lst in by_src.values():
        lst.sort(key=lambda c: c.position)
    per = max(1, k // len(by_src))
    picks: List[Chunk] = []
    for lst in by_src.values():
        if len(lst) <= per:
            picks.extend(lst)
        else:
            step = len(lst) / per
            picks.extend(lst[int(i * step)] for i in range(per))
    return picks[:k]


def numbered(passages: Sequence[Chunk]) -> str:
    out = []
    for i, c in enumerate(passages, start=1):
        loc = []
        if c.page is not None:
            loc.append("page {}".format(c.page))
        if c.section_path:
            loc.append(c.section_path)
        out.append("[Source {}] {}{}\n{}".format(i, c.doc_name, (" | " + ", ".join(loc)) if loc else "",
                                                  c.text.strip()))
    return "\n\n---\n\n".join(out)


def make_cites(markers: Sequence[Any], passages: Sequence[Chunk], claim: str = "") -> List[Cite]:
    out, seen = [], set()
    for m in markers or []:
        try:
            n = int(m)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(passages) and n not in seen:
            seen.add(n)
            c = passages[n - 1]
            out.append(Cite(marker=n, chunk_id=c.chunk_id, source_id=c.doc_id, source_title=c.doc_name,
                            quote=best_quote(claim, c.text) if claim else best_quote(c.text[:200], c.text),
                            page=c.page, slide=c.slide, t_start_ms=c.t_start_ms, section_path=c.section_path))
    return out


# --------------------------------------------------------------------------
# Fact extraction (deterministic generators)
# --------------------------------------------------------------------------

_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
_NUM = re.compile(r"\b\d+(?:[.,]\d+)?%?|\$\d+(?:\.\d\d)?")


def facts(passages: Sequence[Chunk], focus: str = "") -> List[Tuple[int, str]]:
    """(passage number, sentence) for prose sentences, best first."""
    ft = set(tokenize(focus))
    freq = Counter(t for c in passages for t in tokenize(c.text))
    out = []
    for i, c in enumerate(passages, start=1):
        prose = " ".join(ln for ln in c.text.splitlines()
                         if ln.strip() and not re.match(r"\s*(#|\||---)", ln))
        for s in split_sentences(prose):
            s = re.sub(r"^[-*+]\s+|^\d+[.)]\s+", "", s).replace("**", "").strip()
            words = tokenize(s)
            if not (6 <= len(s.split()) <= 45) or len(words) < 4:
                continue
            score = sum(1.0 / (1 + freq[w] ** 0.5) for w in set(words)) + 2.0 * len(ft & set(words))
            out.append((score, i, s))
    out.sort(key=lambda x: -x[0])
    seen, res = set(), []
    for _, i, s in out:
        key = s.lower()[:80]
        if key not in seen:
            seen.add(key)
            res.append((i, s))
    return res


def _key_term(sentence: str, difficulty: str, rng: random.Random) -> Optional[str]:
    nums = _NUM.findall(sentence)
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", sentence) if w.lower() not in STOPWORDS]
    if difficulty == "hard" and nums:
        return nums[0]
    if difficulty == "easy" and words:
        return max(words, key=len)
    if nums and rng.random() < 0.5:
        return nums[0]
    return max(words, key=len) if words else (nums[0] if nums else None)


def _blank(sentence: str, term: str) -> str:
    return re.sub(re.escape(term), "_____", sentence, count=1)


# --------------------------------------------------------------------------
# Deterministic generators
# --------------------------------------------------------------------------


def det_report(passages, guides, params) -> Dict:
    fmt = params.get("format", "briefing")
    fs = facts(passages, params.get("custom_prompt", ""))
    srcs: Dict[str, List[Tuple[int, str]]] = {}
    for i, s in fs:
        srcs.setdefault(passages[i - 1].doc_name, []).append((i, s))
    title = {"briefing": "Briefing", "study_guide": "Study guide", "faq": "FAQ", "timeline": "Timeline",
             "blog_post": "Overview", "custom": "Report"}[fmt]
    sections = []
    if fmt == "faq":
        for name, items in srcs.items():
            topics = guides.get(name, {}).get("topics", []) or [name]
            for t, (i, s) in zip(topics, items[:4]):
                sections.append({"heading": "What does {} say about {}?".format(name, t), "body_md": "{} [{}]".format(s, i)})
    elif fmt == "timeline":
        dated = sorted(((int(_YEAR.search(s).group(1)), i, s) for i, s in fs if _YEAR.search(s)), key=lambda x: x[0])
        rows = ["- **{}** — {} [{}]".format(y, s, i) for y, i, s in dated[:20]] or \
               ["- {} [{}]".format(s, i) for i, s in fs[:10]]
        sections.append({"heading": "Timeline", "body_md": "\n".join(rows)})
    elif fmt == "study_guide":
        terms = []
        for g in guides.values():
            terms.extend(g.get("topics", [])[:5])
        if terms:
            first = fs[0][0] if fs else 1
            sections.append({"heading": "Key terms", "body_md": ", ".join(dict.fromkeys(terms)) + " [{}]".format(first)})
        sections.append({"heading": "Key points", "body_md": "\n".join("- {} [{}]".format(s, i) for i, s in fs[:10])})
        qs = []
        for g in guides.values():
            qs.extend(g.get("questions", [])[:2])
        if qs and fs:
            sections.append({"heading": "Review questions", "body_md": "\n".join(
                "{}. {} (see [{}])".format(n + 1, q, fs[n % len(fs)][0]) for n, q in enumerate(qs[:6]))})
    else:
        if fs:
            sections.append({"heading": "Summary", "body_md": " ".join("{} [{}]".format(s, i) for i, s in fs[:3])})
        for name, items in srcs.items():
            sections.append({"heading": name, "body_md": "\n".join("- {} [{}]".format(s, i) for i, s in items[:4])})
    return {"title": title, "format": fmt, "sections": sections}


def det_data_table(passages, guides, params) -> Dict:
    cols = params.get("columns") or ["Key point", "Topics"]
    if isinstance(cols, str):
        cols = [c.strip() for c in cols.split(",") if c.strip()]
    rows = []
    by_src: Dict[str, List[int]] = {}
    for i, c in enumerate(passages, start=1):
        by_src.setdefault(c.doc_id, []).append(i)
    for n, (sid, idxs) in enumerate(by_src.items()):
        name = passages[idxs[0] - 1].doc_name
        cells = [{"text": name, "citations": [idxs[0]]}]
        sub = [passages[i - 1] for i in idxs]
        for col in cols:
            if col.lower() in ("topics", "topic", "themes"):
                g = guides.get(name, {})
                cells.append({"text": ", ".join(g.get("topics", [])[:5]) or "—", "citations": [idxs[0]]})
                continue
            best = facts(sub, col)
            if best:
                j, s = best[0]
                cells.append({"text": s, "citations": [idxs[j - 1]]})
            else:
                cells.append({"text": "—", "citations": []})
        rows.append({"id": "r{}".format(n + 1), "cells": cells})
    return {"title": params.get("title") or "Data table", "columns": ["Source"] + cols, "rows": rows}


def det_mind_map(passages, guides, params, title) -> Dict:
    fs = facts(passages, params.get("focus", ""))
    counter = [0]

    def nid():
        counter[0] += 1
        return "n{}".format(counter[0])

    root = {"id": nid(), "label": (params.get("focus") or title)[:80], "citations": [1], "children": []}
    by_src: Dict[str, List[int]] = {}
    for i, c in enumerate(passages, start=1):
        by_src.setdefault(c.doc_name, []).append(i)
    for name, idxs in by_src.items():
        snode = {"id": nid(), "label": name, "citations": [idxs[0]], "children": []}
        topics = guides.get(name, {}).get("topics", [])[:6]
        if not topics:  # no guide yet: frequent terms of this source's passages
            from app.core.guide import _top_terms

            topics = _top_terms(" ".join(passages[i - 1].text for i in idxs), 6)
        src_facts = [(i, s) for i, s in fs if i in idxs]
        used = set()
        for t in topics:
            tw = set(tokenize(t))
            hits = [(i, s) for i, s in src_facts if tw & set(tokenize(s)) and s not in used][:3]
            if not hits:
                continue
            tnode = {"id": nid(), "label": t, "citations": [hits[0][0]], "children": []}
            for i, s in hits:
                used.add(s)
                tnode["children"].append({"id": nid(), "label": (s[:77] + "…") if len(s) > 80 else s,
                                          "citations": [i], "children": []})
            snode["children"].append(tnode)
        if not snode["children"]:
            for i, s in src_facts[:4]:
                snode["children"].append({"id": nid(), "label": (s[:77] + "…") if len(s) > 80 else s,
                                          "citations": [i], "children": []})
        root["children"].append(snode)
    # Pad toward the minimum size with further facts, round-robin by source.
    if _count(root) < MIND_MAP_MIN_NODES:
        used = {n["label"] for s_ in root["children"] for t in s_["children"] for n in t["children"]}
        spare = [(i, s) for i, s in fs if ((s[:77] + "…") if len(s) > 80 else s) not in used]
        srcs = root["children"]
        k = 0
        while _count(root) < MIND_MAP_MIN_NODES and spare and srcs:
            i, s = spare.pop(0)
            target = next((sn for sn in srcs if sn["label"] == passages[i - 1].doc_name), srcs[k % len(srcs)])
            target["children"].append({"id": nid(), "label": (s[:77] + "…") if len(s) > 80 else s,
                                       "citations": [i], "children": []})
            k += 1
    return {"title": title, "root": root}


def det_flashcards(passages, guides, params, seed) -> Dict:
    n = QUANTITY.get(params.get("quantity", "standard"), 20)
    diff = params.get("difficulty", "medium")
    rng = random.Random(seed)
    cards = []
    for i, s in facts(passages, params.get("focus", "")):
        term = _key_term(s, diff, rng)
        if not term:
            continue
        front = _blank(s, term)
        if front == s:
            continue
        cards.append({"id": "c{}".format(len(cards) + 1), "front": front, "back": "{} — {}".format(term, s),
                      "citations": [i]})
        if len(cards) >= n:
            break
    return {"title": "Flashcards", "difficulty": diff, "cards": cards}


def det_quiz(passages, guides, params, seed) -> Dict:
    n = QUANTITY.get(params.get("quantity", "standard"), 20)
    types = [t for t in params.get("types") or ["multiple_choice"] if t in QUIZ_TYPES] or ["multiple_choice"]
    diff = params.get("difficulty", "medium")
    rng = random.Random(seed)
    fs = facts(passages, params.get("focus", ""))
    terms = [(i, s, _key_term(s, diff, rng)) for i, s in fs]
    terms = [(i, s, t) for i, s, t in terms if t]
    pool_nums = [t for _, _, t in terms if _NUM.fullmatch(t)]
    pool_words = [t for _, _, t in terms if not _NUM.fullmatch(t)]
    items = []
    for k, (i, s, term) in enumerate(terms):
        qtype = types[k % len(types)]
        q_id = "q{}".format(len(items) + 1)
        if qtype in ("multiple_choice",):
            pool = [p for p in (pool_nums if _NUM.fullmatch(term) else pool_words) if p.lower() != term.lower()]
            distract = list(dict.fromkeys(pool))
            rng.shuffle(distract)
            if len(distract) < 3:
                continue
            opts = distract[:3] + [term]
            rng.shuffle(opts)
            items.append({"id": q_id, "type": qtype, "question": _blank(s, term), "options": opts,
                          "answer": opts.index(term), "rationale": s, "citations": [i]})
        elif qtype == "multi_select":
            others = [x for x in terms if x[0] != i or x[1] != s]
            if len(others) < 3:
                continue
            true2 = [s, others[0][1]]
            falsify = []
            for oi, os_, ot in others[1:3]:
                swap = next((w for w in (pool_words + pool_nums) if w.lower() != ot.lower()), None)
                if swap:
                    falsify.append(os_.replace(ot, swap, 1))
            if len(falsify) < 2:
                continue
            opts = true2 + falsify
            order = list(range(4))
            rng.shuffle(order)
            opts = [opts[j] for j in order]
            answer = sorted([order.index(0), order.index(1)])
            items.append({"id": q_id, "type": qtype, "question": "Which statements are supported by the sources?",
                          "options": opts, "answer": answer, "rationale": "Supported: “{}” and “{}”".format(*true2),
                          "citations": [i, others[0][0]]})
        elif qtype == "fill_blank":
            items.append({"id": q_id, "type": qtype, "question": _blank(s, term), "answer": term,
                          "rationale": s, "citations": [i]})
        else:  # short_answer
            items.append({"id": q_id, "type": qtype,
                          "question": "In your own words: {}".format(_blank(s, term).rstrip(".") + "?"),
                          "answer": s, "rationale": s, "citations": [i]})
        if len(items) >= n:
            break
    return {"title": "Quiz", "difficulty": diff, "items": items}


# --------------------------------------------------------------------------
# LLM generators (JSON schemas, untrusted-data rule in every prompt)
# --------------------------------------------------------------------------

_CITE_ARR = {"type": "array", "items": {"type": "integer"}}

SCHEMAS: Dict[str, Dict] = {
    "report": {"type": "object", "additionalProperties": False, "required": ["title", "sections"],
               "properties": {"title": {"type": "string"}, "sections": {"type": "array", "items": {
                   "type": "object", "additionalProperties": False, "required": ["heading", "body_md"],
                   "properties": {"heading": {"type": "string"}, "body_md": {"type": "string"}}}}}},
    "data_table": {"type": "object", "additionalProperties": False, "required": ["title", "columns", "rows"],
                   "properties": {"title": {"type": "string"}, "columns": {"type": "array", "items": {"type": "string"}},
                                  "rows": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                           "required": ["cells"], "properties": {"cells": {"type": "array", "items": {
                                               "type": "object", "additionalProperties": False,
                                               "required": ["text", "citations"],
                                               "properties": {"text": {"type": "string"}, "citations": _CITE_ARR}}}}}}}},
    "mind_map": {"type": "object", "additionalProperties": False, "required": ["title", "nodes"],
                 "properties": {"title": {"type": "string"}, "nodes": {"type": "array", "items": {
                     "type": "object", "additionalProperties": False, "required": ["id", "label", "parent", "citations"],
                     "properties": {"id": {"type": "string"}, "label": {"type": "string"},
                                    "parent": {"type": "string"}, "citations": _CITE_ARR}}}}},
    "flashcards": {"type": "object", "additionalProperties": False, "required": ["cards"],
                   "properties": {"cards": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                  "required": ["front", "back", "citations"],
                                  "properties": {"front": {"type": "string"}, "back": {"type": "string"},
                                                 "citations": _CITE_ARR}}}}},
    "quiz": {"type": "object", "additionalProperties": False, "required": ["items"],
             "properties": {"items": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                            "required": ["type", "question", "options", "answer", "rationale", "citations"],
                            "properties": {"type": {"type": "string", "enum": list(QUIZ_TYPES)},
                                           "question": {"type": "string"},
                                           "options": {"type": "array", "items": {"type": "string"}},
                                           "answer": {"type": "string",
                                                      "description": "MC: option index; multi-select: comma-separated indexes; else the answer text"},
                                           "rationale": {"type": "string"}, "citations": _CITE_ARR}}}}},
}

_BASE_RULES = (
    "You create study and research material for a notebook. Use ONLY the numbered SOURCE "
    "passages; never add outside facts. Every item must cite the number(s) of the passage(s) "
    "that support it; an item you cannot support with a passage must be left out. "
    + UNTRUSTED_DATA_RULE
)

_TASK = {
    "report": "Write a {fmt} report{length}. In body_md (markdown), end every sentence that states a "
              "fact with bracketed passage numbers, e.g. 'Exams are closed-book [3].'{custom}",
    "data_table": "Build a data table. Rows and columns: {describe}. Each cell's citations list the "
                  "passage numbers supporting that cell (empty only for a header-like label).",
    "mind_map": "Build a mind map of the sources{focus}: a single root (parent \"\"), at most 4 levels "
                "deep, 20 to 80 nodes, short labels (under 8 words). Every node cites passages.",
    "flashcards": "Write {n} flashcards at {diff} difficulty{focus}. Front: a question or cue. Back: the "
                  "answer, concise.",
    "quiz": "Write {n} quiz questions at {diff} difficulty{focus}, using only these types: {types}. "
            "multiple_choice: 4 options, answer = index of the correct option. multi_select: 4 options, "
            "answer = comma-separated indexes of all correct ones. fill_blank: question has '_____', answer "
            "is the missing text. short_answer: answer is a model answer. rationale explains why, from "
            "the sources.",
}


def build_prompt(atype: str, params: Dict, guides: Dict[str, Dict], passages) -> Tuple[str, str]:
    n = QUANTITY.get(params.get("quantity", "standard"), 20)
    focus = params.get("focus") or params.get("custom_prompt") or ""
    task = _TASK[atype].format(
        fmt=params.get("format", "briefing").replace("_", " "),
        length={"short": " of about 300 words", "long": " of about 1500 words"}.get(params.get("length", ""), ""),
        custom=(" Follow this brief: " + params["custom_prompt"]) if params.get("custom_prompt") else "",
        describe=params.get("describe") or ", ".join(params.get("columns") or []) or "one row per source; useful columns",
        focus=(" focused on: " + focus) if focus else "",
        n=n, diff=params.get("difficulty", "medium"),
        types=", ".join(params.get("types") or ["multiple_choice"]),
    )
    if params.get("language"):
        task += " Write in {}.".format(params["language"])
    system = _BASE_RULES + "\n\nTASK: " + task
    guide_txt = "\n".join("- {}: {}".format(name, re.sub(r"\s*\[\d+\]", "", g.get("summary", ""))[:600])
                          for name, g in guides.items() if g.get("summary"))
    user = ("SOURCE GUIDES (summaries, not citable):\n{}\n\nSOURCE PASSAGES (cite these numbers):\n{}"
            .format(guide_txt or "(none)", numbered(passages)))
    return system, user


def llm_generate(atype: str, params: Dict, guides, passages, llm) -> Dict:
    system, user = build_prompt(atype, params, guides, passages)
    data, _ = llm.complete_json(system, [{"role": "user", "content": user}], SCHEMAS[atype], max_tokens=8000)
    data["_prompt"] = system
    return data


# --------------------------------------------------------------------------
# Validation: map citations, drop unsupported items, enforce bounds
# --------------------------------------------------------------------------


def _depth(node: Dict, d: int = 0) -> int:
    return max([d] + [_depth(c, d + 1) for c in node.get("children", [])])


def _count(node: Dict) -> int:
    return 1 + sum(_count(c) for c in node.get("children", []))


def _tree_from_flat(nodes: List[Dict]) -> Dict:
    by_id = {str(n["id"]): dict(n, children=[]) for n in nodes}
    root = None
    for n in by_id.values():
        p = str(n.get("parent") or "")
        if p and p in by_id and p != n["id"]:
            by_id[p]["children"].append(n)
        elif root is None:
            root = n
    return root or {"id": "root", "label": "Sources", "citations": [], "children": []}


def _prune(node: Dict, depth: int, budget: List[int]) -> Optional[Dict]:
    if budget[0] <= 0:
        return None
    budget[0] -= 1
    kids = []
    if depth < MIND_MAP_MAX_DEPTH:
        for c in node.get("children", []):
            pc = _prune(c, depth + 1, budget)
            if pc is not None:
                kids.append(pc)
    node = dict(node, children=kids)
    return node


def finalize(atype: str, raw: Dict, passages: Sequence[Chunk], params: Dict, embedder=None,
             support_floor: float = 0.5) -> Tuple[BaseModel, int]:
    """Validate raw generator output. Returns (model, dropped_item_count)."""
    dropped = 0
    if atype == "report":
        sections = []
        for s in raw.get("sections", []):
            from app.core.retrieval import SearchHit

            hits = [SearchHit(c, 0, 0) for c in passages]
            cited = process_citations(s.get("body_md", ""), hits, embedder, support_floor)
            cites = make_cites([c.marker for c in cited.citations], passages, s.get("body_md", ""))
            if not cites:
                dropped += 1
                continue
            sections.append(ReportSection(heading=str(s.get("heading", "")).strip() or "Section",
                                          body_md=cited.text, citations=cites))
        return Report(title=raw.get("title") or "Report", format=params.get("format", "briefing"),
                      sections=sections), dropped
    if atype == "data_table":
        rows = []
        for n, r in enumerate(raw.get("rows", [])):
            cells = [Cell(text=str(c.get("text", "")), citations=make_cites(c.get("citations"), passages, c.get("text", "")))
                     for c in r.get("cells", [])]
            row_cites = []
            seen = set()
            for c in cells:
                for ct in c.citations:
                    if ct.marker not in seen:
                        seen.add(ct.marker)
                        row_cites.append(ct)
            if not row_cites:
                dropped += 1
                continue
            rows.append(TableRow(id=r.get("id") or "r{}".format(n + 1), cells=cells, citations=row_cites))
        return DataTable(title=raw.get("title") or "Data table", columns=[str(c) for c in raw.get("columns", [])],
                         rows=rows), dropped
    if atype == "mind_map":
        root = raw["root"] if "root" in raw else _tree_from_flat(raw.get("nodes", []))
        root = _prune(root, 0, [MIND_MAP_MAX_NODES])

        def build(node: Dict) -> Optional[MindNode]:
            nonlocal dropped
            kids = [k for k in (build(c) for c in node.get("children", [])) if k is not None]
            cites = make_cites(node.get("citations"), passages, node.get("label", ""))
            if not cites and kids:
                cites = kids[0].citations[:1]
            if not cites:
                dropped += 1
                return None
            return MindNode(id=str(node.get("id")), label=" ".join(str(node.get("label", "")).split())[:120],
                            citations=cites, children=kids)

        mroot = build(root) or MindNode(id="root", label=raw.get("title", "Sources"),
                                         citations=make_cites([1], passages), children=[])
        return MindMap(title=raw.get("title") or mroot.label, root=mroot), dropped
    if atype == "flashcards":
        cards = []
        for c in raw.get("cards", []):
            cites = make_cites(c.get("citations"), passages, c.get("back", ""))
            if not cites or not str(c.get("front", "")).strip():
                dropped += 1
                continue
            cards.append(Flashcard(id=c.get("id") or "c{}".format(len(cards) + 1), front=str(c["front"]).strip(),
                                   back=str(c.get("back", "")).strip(), citations=cites))
        return Flashcards(title=raw.get("title") or "Flashcards", difficulty=params.get("difficulty", "medium"),
                          cards=cards), dropped
    if atype == "quiz":
        items = []
        for it in raw.get("items", []):
            cites = make_cites(it.get("citations"), passages, it.get("rationale", ""))
            qtype = it.get("type")
            opts = [str(o) for o in it.get("options") or []]
            ans = it.get("answer")
            try:
                if qtype == "multiple_choice":
                    ans = int(ans)
                    ok = 0 <= ans < len(opts) and len(opts) >= 2
                elif qtype == "multi_select":
                    ans = sorted({int(x) for x in (ans if isinstance(ans, list) else str(ans).split(",")) if str(x).strip()})
                    ok = bool(ans) and all(0 <= x < len(opts) for x in ans)
                else:
                    ans = str(ans)
                    ok = bool(ans.strip())
            except (TypeError, ValueError):
                ok = False
            if not cites or qtype not in QUIZ_TYPES or not ok:
                dropped += 1
                continue
            items.append(QuizItem(id=it.get("id") or "q{}".format(len(items) + 1), type=qtype,
                                  question=str(it.get("question", "")), options=opts, answer=ans,
                                  rationale=str(it.get("rationale", "")), citations=cites))
        return Quiz(title=raw.get("title") or "Quiz", difficulty=params.get("difficulty", "medium"),
                    items=items), dropped
    raise ValueError(atype)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def generate(atype: str, params: Dict, store, source_ids: Optional[List[str]], settings, llm=None,
             title: str = "Notebook", progress=None, cancelled=None) -> Dict:
    """Run the full pipeline. Returns the artifact record fields."""
    if atype not in TYPES:
        raise ValueError("Unknown artifact type {!r}".format(atype))
    focus = (params.get("focus") or params.get("custom_prompt") or "").strip()
    scope = ScopeFilter.parse(doc_ids=source_ids) if source_ids else None
    if focus:
        hits = store.index.search(focus, top_k=settings.top_k, scope=scope)
        res = classify(focus, hits, relevance_floor=settings.relevance_floor, corpus_empty=store.is_empty,
                       scope=scope, scope_matched_documents=store.count_matching(scope),
                       dense_rescue_floor=settings.dense_rescue_floor)
        if not res.allows_llm or res.decision in ("small_talk", "meta"):
            raise StudioRefused("The focus “{}” is outside what the selected sources cover ({}).".format(
                focus, res.reason))
    passages = select_passages(store, source_ids, focus)
    if not passages:
        raise ValueError("No ready sources are selected.")
    if progress:
        progress(0.2)
    guides = {}
    for d in store.list_documents():
        if source_ids and d.doc_id not in source_ids:
            continue
        src = store.get_source(d.doc_id)
        guides[d.name] = (src.guide if src else {}) or {}
    seed = int(hashlib.sha256(json.dumps([atype, params, [c.chunk_id for c in passages]],
                                         sort_keys=True).encode()).hexdigest()[:8], 16)
    raw, generated_by, prompt = None, "extractive", "deterministic generator (no LLM configured)"
    if cancelled and cancelled():
        raise InterruptedError()
    if llm is not None:
        try:
            raw = llm_generate(atype, params, guides, passages, llm)
            prompt = raw.pop("_prompt", "")
            generated_by = "llm"
            if atype == "quiz":
                for it in raw.get("items", []):
                    if it.get("type") == "multi_select" and isinstance(it.get("answer"), str):
                        it["answer"] = [x for x in it["answer"].split(",") if x.strip()]
        except (LLMUnavailableError, LLMRefusalError, ValueError, KeyError) as exc:
            logger.warning("Studio LLM generation failed (%s); using deterministic generator", exc)
            raw = None
    if raw is None:
        raw = {"report": lambda: det_report(passages, guides, params),
               "data_table": lambda: det_data_table(passages, guides, params),
               "mind_map": lambda: det_mind_map(passages, guides, params, title),
               "flashcards": lambda: det_flashcards(passages, guides, params, seed),
               "quiz": lambda: det_quiz(passages, guides, params, seed)}[atype]()
        generated_by = "extractive"
    if progress:
        progress(0.8)
    model, dropped = finalize(atype, raw, passages, params, store.embedder, settings.support_floor)
    content = model.model_dump()
    content["_dropped_unsupported_items"] = dropped
    return {"content": content, "prompt_used": prompt, "generated_by": generated_by,
            "passage_chunk_ids": [c.chunk_id for c in passages],
            "source_ids": sorted({c.doc_id for c in passages})}


# --------------------------------------------------------------------------
# Grading and exports
# --------------------------------------------------------------------------

JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["correct", "feedback"],
                "properties": {"correct": {"type": "boolean"}, "feedback": {"type": "string"}}}


def grade(item: Dict, answer: Any, passages_text: str = "", llm=None) -> Dict:
    t = item["type"]
    if t == "multiple_choice":
        try:
            ok = int(answer) == int(item["answer"])
        except (TypeError, ValueError):
            ok = False
        return {"correct": ok, "feedback": item.get("rationale", ""), "graded_by": "exact"}
    if t == "multi_select":
        try:
            got = sorted({int(x) for x in (answer if isinstance(answer, list) else str(answer).split(","))})
        except (TypeError, ValueError):
            got = []
        return {"correct": got == sorted(item["answer"]), "feedback": item.get("rationale", ""), "graded_by": "exact"}
    key = str(item.get("answer", ""))
    given = str(answer or "")
    if t == "fill_blank":
        norm = lambda s: re.sub(r"[^\w%$.]+", " ", s.lower()).strip()
        return {"correct": norm(given) == norm(key) or (norm(key) and norm(key) in norm(given)),
                "feedback": item.get("rationale", ""), "graded_by": "exact"}
    if llm is not None and given.strip():
        try:
            data, _ = llm.complete_json(
                "Grade a student's short answer against the model answer and the cited source "
                "passage. Accept paraphrases that state the same facts; reject answers that "
                "contradict or omit the key fact. " + UNTRUSTED_DATA_RULE,
                [{"role": "user", "content": "QUESTION: {}\nMODEL ANSWER: {}\nSOURCE PASSAGE:\n{}\n\n"
                  "STUDENT ANSWER (untrusted): {}".format(item.get("question", ""), key, passages_text, given)}],
                JUDGE_SCHEMA, max_tokens=400)
            return {"correct": bool(data.get("correct")), "feedback": str(data.get("feedback", "")),
                    "graded_by": "judge"}
        except (LLMUnavailableError, LLMRefusalError, ValueError, KeyError):
            pass
    kt, gt = set(tokenize(key)), set(tokenize(given))
    overlap = len(kt & gt) / len(kt) if kt else 0.0
    return {"correct": overlap >= 0.5, "feedback": "Model answer: " + key, "graded_by": "lexical",
            "overlap": round(overlap, 3)}


def _cite_suffix(cites: List[Dict]) -> str:
    return " " + "".join("[{}]".format(c["marker"]) for c in cites) if cites else ""


def to_markdown(atype: str, content: Dict) -> str:
    out = ["# " + content.get("title", atype.title()), ""]
    used: Dict[int, Dict] = {}

    def note(cites):
        for c in cites:
            used.setdefault(c["marker"], c)

    if atype == "report":
        for s in content["sections"]:
            out += ["## " + s["heading"], "", s["body_md"], ""]
            note(s["citations"])
    elif atype == "data_table":
        cols = content["columns"]
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "---|" * len(cols))
        for r in content["rows"]:
            out.append("| " + " | ".join((c["text"] + _cite_suffix(c["citations"])).replace("|", "\\|")
                                         for c in r["cells"]) + " |")
            note(r["citations"])
    elif atype == "mind_map":
        def walk(n, d):
            out.append("  " * d + "- " + n["label"] + _cite_suffix(n["citations"]))
            note(n["citations"])
            for c in n["children"]:
                walk(c, d + 1)
        walk(content["root"], 0)
    elif atype == "flashcards":
        for i, c in enumerate(content["cards"], 1):
            out += ["**{}. {}**".format(i, c["front"]), "", c["back"] + _cite_suffix(c["citations"]), ""]
            note(c["citations"])
    elif atype == "quiz":
        for i, q in enumerate(content["items"], 1):
            out.append("**{}. {}**".format(i, q["question"]))
            for j, o in enumerate(q.get("options") or []):
                out.append("   {}. {}".format("ABCDEFGH"[j], o))
            ans = q["answer"]
            if q["type"] == "multiple_choice":
                ans = "ABCDEFGH"[int(ans)]
            elif q["type"] == "multi_select":
                ans = ", ".join("ABCDEFGH"[int(a)] for a in ans)
            out += ["", "_Answer:_ {}  ".format(ans), "_Why:_ " + q["rationale"] + _cite_suffix(q["citations"]), ""]
            note(q["citations"])
    if used:
        out += ["", "## Sources", ""]
        for m in sorted(used):
            c = used[m]
            loc = " (p. {})".format(c["page"]) if c.get("page") else ""
            out.append("[{}] {}{}: “{}”".format(m, c["source_title"], loc, c.get("quote", "")))
    return "\n".join(out).strip() + "\n"


def to_csv(content: Dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(content["columns"])
    for r in content["rows"]:
        w.writerow([c["text"] for c in r["cells"]])
    return buf.getvalue()


def citations_csv(content: Dict) -> str:
    """One line per (row, column, citation) - the 'citations sheet'."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["row", "column", "marker", "source", "page", "quote"])
    for ri, r in enumerate(content["rows"], 1):
        for ci, c in enumerate(r["cells"]):
            col = content["columns"][ci] if ci < len(content["columns"]) else str(ci)
            for ct in c["citations"]:
                w.writerow([ri, col, ct["marker"], ct["source_title"], ct.get("page") or "", ct.get("quote", "")])
    return buf.getvalue()


def to_docx(md: str) -> bytes:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise ValueError("DOCX export requires python-docx.") from exc
    d = docx.Document()
    for line in md.splitlines():
        if line.startswith("# "):
            d.add_heading(line[2:], level=1)
        elif line.startswith("## "):
            d.add_heading(line[3:], level=2)
        elif line.strip().startswith(("- ", "* ")):
            d.add_paragraph(line.strip()[2:].replace("**", ""), style="List Bullet")
        elif line.strip():
            d.add_paragraph(line.replace("**", "").replace("_", ""))
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def to_html(md: str, title: str) -> str:
    import html as _h

    body = []
    for line in md.splitlines():
        e = _h.escape(line)
        if line.startswith("# "):
            body.append("<h1>{}</h1>".format(e[2:]))
        elif line.startswith("## "):
            body.append("<h2>{}</h2>".format(e[3:]))
        elif line.strip().startswith("- "):
            body.append("<li>{}</li>".format(e.strip()[2:]))
        elif line.strip():
            body.append("<p>{}</p>".format(e))
    return ("<!doctype html><html><head><meta charset='utf-8'><title>{}</title>"
            "<style>body{{font-family:sans-serif;max-width:46em;margin:2em auto;line-height:1.5}}</style>"
            "</head><body>{}</body></html>").format(_h.escape(title), "\n".join(body))


def to_pdf(md: str, title: str) -> Optional[bytes]:
    try:
        from weasyprint import HTML  # optional
    except Exception:
        return None
    return HTML(string=to_html(md, title)).write_pdf()


def mind_map_svg(content: Dict) -> str:
    """Static tidy-tree SVG (left to right), labels escaped."""
    import html as _h

    rows: List[Tuple[int, int, str, Optional[int]]] = []  # (depth, y, label, parent_index)

    def walk(n, d, parent):
        idx = len(rows)
        rows.append((d, len(rows), n["label"], parent))
        for c in n["children"]:
            walk(c, d + 1, idx)
    walk(content["root"], 0, None)
    W, H, dx, dy = 1100, max(120, 26 * len(rows) + 40), 260, 26
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" font-family="sans-serif" font-size="12">'
             .format(W, H)]
    pos = [(20 + d * dx, 30 + y * dy) for d, y, _, _ in rows]
    for i, (d, y, label, p) in enumerate(rows):
        if p is not None:
            (x1, y1), (x2, y2) = pos[p], pos[i]
            parts.append('<path d="M{} {} C{} {} {} {} {} {}" fill="none" stroke="#9aa4b2"/>'.format(
                x1 + 8, y1, x1 + dx / 2, y1, x1 + dx / 2, y2, x2 - 4, y2))
    for i, (d, y, label, p) in enumerate(rows):
        x, yy = pos[i]
        lab = label if len(label) <= 40 else label[:39] + "…"
        parts.append('<circle cx="{}" cy="{}" r="4" fill="#2a56c6"/><text x="{}" y="{}">{}</text>'.format(
            x, yy, x + 8, yy + 4, _h.escape(lab)))
    parts.append("</svg>")
    return "".join(parts)
