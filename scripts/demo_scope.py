#!/usr/bin/env python3
"""Demonstrate scope enforcement without an API key or any network access.

Loads the two example corpora and runs a labelled set of questions through the
retrieval gate, showing which are accepted and which are refused — and, for the
cross-domain cases, that each bot stays inside its own subject.

    python scripts/demo_scope.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.guard import ScopeDecision, classify  # noqa: E402
from app.core.store import DocumentStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FLOOR = 0.40

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
)

LABEL = {
    ScopeDecision.IN_SCOPE: (GREEN, "ANSWER   "),
    ScopeDecision.BORDERLINE: (YELLOW, "VERIFY   "),
    ScopeDecision.OUT_OF_SCOPE: (RED, "REFUSE   "),
    ScopeDecision.SMALL_TALK: (DIM, "GREETING "),
    ScopeDecision.META: (DIM, "META     "),
    ScopeDecision.NO_CORPUS: (RED, "NO DOCS  "),
}


def build(path: Path) -> DocumentStore:
    store = DocumentStore(data_dir=Path(tempfile.mkdtemp()))
    store.add(path.name, path.read_bytes())
    return store


def run(title: str, store: DocumentStore, questions) -> None:
    print("\n{}{}{}".format(BOLD, title, RESET))
    print(DIM + "-" * 74 + RESET)
    for question in questions:
        hits = store.index.search(question, top_k=6)
        result = classify(question, hits, FLOOR, store.is_empty)
        color, label = LABEL[result.decision]
        print("  {}{}{}  {:.3f}  {}".format(
            color, label, RESET, result.best_score, question))


def main() -> int:
    syllabus = build(ROOT / "examples" / "course_ta" / "cs1428_syllabus.md")
    support = build(ROOT / "examples" / "customer_care" / "returns_and_shipping.md")

    print("\n{}Scope enforcement demo{}".format(BOLD, RESET))
    print("{}No API key required — the gate is pure retrieval logic.{}".format(
        DIM, RESET))
    print("{}Relevance floor: {:.2f}{}".format(DIM, FLOOR, RESET))

    run("COURSE TA  ·  questions about the syllabus (should answer)", syllabus, [
        "How is the final grade calculated?",
        "What is the late work policy?",
        "When are office hours?",
        "Can I use AI tools on my homework?",
        "What happens if I miss an exam?",
    ])

    run("COURSE TA  ·  general knowledge (should refuse)", syllabus, [
        "What is the capital of France?",
        "Give me a recipe for lasagna.",
        "Who won the World Series in 2019?",
        "Tell me a joke about horses.",
        "Should I buy Tesla stock?",
    ])

    run("COURSE TA  ·  another bot's domain (should not answer)", syllabus, [
        "How much is overnight shipping?",
        "What is your warranty on jackets?",
        "Can I return these boots?",
    ])

    run("SUPPORT BOT  ·  questions about the policy (should answer)", support, [
        "How long do I have to return an item?",
        "How much is expedited shipping?",
        "What does the warranty cover?",
        "Can I cancel my order?",
    ])

    run("SUPPORT BOT  ·  the course bot's domain (should not answer)", support, [
        "When is the final exam?",
        "Who is the course instructor?",
        "What chapters does Exam 1 cover?",
    ])

    run("CONVERSATIONAL  ·  handled without retrieval", syllabus, [
        "hello",
        "thanks",
        "What can you do?",
    ])

    print("\n{}{}  = answered from documents   {}{}  = sent to the model to verify"
          "   {}{}  = refused before any LLM call{}\n".format(
              GREEN, "ANSWER", YELLOW, "VERIFY", RED, "REFUSE", RESET))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
