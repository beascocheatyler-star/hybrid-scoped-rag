#!/usr/bin/env python3
"""A guided tour of hybrid scoped retrieval. No API key, no network, no cost.

    python scripts/demo_hybrid.py

Shows, in order:
  1. Tagging documents with metadata.
  2. Hybrid retrieval, and which retriever found each passage.
  3. Scoping: the same question answered differently under different filters.
  4. The scope gate refusing an off-topic question before any LLM call.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.core.embedding import HashingEmbedder  # noqa: E402
from app.core.guard import RELEVANCE_FLOOR_DEFAULT, classify  # noqa: E402
from app.core.metadata import ScopeFilter  # noqa: E402
from app.core.store import DocumentStore  # noqa: E402

EXAMPLES = REPO_ROOT / "examples"


def heading(text):
    print("\n{}\n{}".format(text, "=" * len(text)))


def show_hits(hits):
    if not hits:
        print("   (nothing retrieved)")
        return
    for hit in hits:
        found = "+".join(hit.retrievers) or "none"
        ranks = "lex #{} / dense #{}".format(
            hit.lexical_rank if hit.lexical_rank else "-",
            hit.dense_rank if hit.dense_rank else "-",
        )
        print(
            "   {:28s} {:11s} {:18s} lex={:.3f} dense={:.3f}".format(
                hit.chunk.doc_name[:28], found, ranks, hit.normalized, hit.dense_score
            )
        )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        store = DocumentStore(Path(tmp), embedder=HashingEmbedder())

        heading("1. Ingest and tag")
        store.add(
            "cs1428_syllabus.md",
            (EXAMPLES / "course_ta" / "cs1428_syllabus.md").read_bytes(),
            metadata={"course": "CS1428", "term": "Fall2025", "kind": "syllabus"},
        )
        store.add(
            "returns_and_shipping.md",
            (EXAMPLES / "customer_care" / "returns_and_shipping.md").read_bytes(),
            metadata={"dept": "support", "kind": "policy"},
        )
        for doc in store.list_documents():
            print("   {:28s} {:2d} passages  {}".format(
                doc.name, doc.chunk_count, doc.metadata
            ))

        print("\n   Facets available for filtering:")
        for key, values in sorted(store.facets().items()):
            print("     {:8s} {}".format(key, dict(values)))

        heading("2. Hybrid retrieval")
        for question in [
            "When is the final exam?",
            "How do I return a pair of boots?",
        ]:
            print("\n   {!r}".format(question))
            show_hits(store.index.search(question, top_k=3))

        print("\n   'both' means BM25 and the dense retriever independently")
        print("   surfaced that passage — the agreement RRF rewards.")

        heading("3. Scoping")
        question = "What is the policy on returns?"
        print("   {!r}\n".format(question))

        for label, scope in [
            ("no filter", None),
            ("course=CS1428", ScopeFilter.parse({"course": "CS1428"})),
            ("dept=support", ScopeFilter.parse({"dept": "support"})),
            ("kind in (syllabus, policy)",
             ScopeFilter.parse({"kind": ["syllabus", "policy"]})),
        ]:
            matched = store.count_matching(scope)
            hits = store.index.search(question, top_k=2, scope=scope)
            names = sorted({h.chunk.doc_name for h in hits}) or ["(none)"]
            print("   {:28s} {} doc(s) -> {}".format(label, matched, ", ".join(names)))

        print("\n   The filter runs before scoring, so excluded documents are")
        print("   never candidates — not merely ranked lower.")

        heading("4. The scope gate")
        for question in [
            "When is the final exam?",
            "Who won the Super Bowl in 1998?",
            "hello",
        ]:
            hits = store.index.search(question, top_k=6)
            result = classify(
                question, hits, RELEVANCE_FLOOR_DEFAULT, store.is_empty
            )
            print("   {:38s} {:14s} (relevance {:.3f})".format(
                repr(question)[:38], result.decision, result.best_score
            ))

        print("\n   Decisions that are not in_scope/borderline/small_talk/meta")
        print("   are refused deterministically — no token is ever spent.")

        heading("Next")
        print("   python scripts/measure_hybrid.py   reproduce the tuning numbers")
        print("   python run.py                      start the web UI")
        print("   pytest                             run the test suite")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
