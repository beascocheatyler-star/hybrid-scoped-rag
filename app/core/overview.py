"""Notebook overview: a summary of all sources plus 3-5 suggested questions.

Built from the per-source guides (already map-reduced), so it is cheap: one
LLM call over at most ~40 guide summaries, or a deterministic merge without
an LLM. Stored in ``notebooks.overview_json`` with a snapshot of the ready
source ids; the API reports it as ``stale`` once the source set changes, and
a regeneration job is queued when sources are added or removed.
"""

from __future__ import annotations

import re
from typing import Dict, List

from app.core.llm import UNTRUSTED_DATA_RULE, LLMRefusalError, LLMUnavailableError

SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"},
                   "questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["summary", "questions"],
    "additionalProperties": False,
}
_MARK = re.compile(r"\s*\[\d{1,3}\]")


def _first_sentence(text: str) -> str:
    t = _MARK.sub("", text).strip()
    m = re.search(r"(.+?[.!?])(\s|$)", t)
    return (m.group(1) if m else t)[:300]


def build_overview(sources: List[Dict], llm=None) -> Dict:
    """sources: [{"id", "title", "guide"}] for ready sources."""
    ids = [s["id"] for s in sources]
    if not sources:
        return {"summary": "", "questions": [], "source_ids": [], "generated_by": "none"}
    if llm is not None:
        listing = "\n\n".join("SOURCE: {}\n{}".format(s["title"], _MARK.sub("", s["guide"].get("summary", ""))[:1500])
                              for s in sources[:40])
        try:
            data, _ = llm.complete_json(
                "Write a 100-180 word overview of what this research notebook's sources "
                "cover, naming the sources, then 3-5 questions a reader could ask that "
                "the sources answer. Use only the summaries. " + UNTRUSTED_DATA_RULE,
                [{"role": "user", "content": listing}], SCHEMA, max_tokens=1200)
            qs = [str(q).strip() for q in data.get("questions", []) if str(q).strip()][:5]
            if data.get("summary"):
                return {"summary": str(data["summary"]).strip(), "questions": qs,
                        "source_ids": ids, "generated_by": "llm"}
        except (LLMUnavailableError, LLMRefusalError, ValueError, KeyError):
            pass
    parts = ["This notebook has {} source{}.".format(len(sources), "" if len(sources) == 1 else "s")]
    for s in sources[:6]:
        first = _first_sentence(s["guide"].get("summary", ""))
        if first:
            parts.append("{}: {}".format(s["title"], first))
    questions: List[str] = []
    for s in sources:
        for q in s["guide"].get("questions", [])[:2]:
            if q not in questions:
                questions.append(q)
    return {"summary": " ".join(parts), "questions": questions[:5], "source_ids": ids,
            "generated_by": "extractive"}
