#!/usr/bin/env python3
"""Measure the scope gate against a labelled question set.

Run this after changing the retrieval logic, the scoring formula, or the example
corpora, to see where the decision boundary actually falls. Prints the score
distribution and the error count at a range of thresholds.

    python scripts/measure_scope.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.guard import relevance  # noqa: E402
from app.core.store import DocumentStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

SYLLABUS_IN = [
    "How is the final grade calculated?",
    "What is the late work policy?",
    "When are office hours?",
    "What happens if I miss an exam?",
    "How much does the zyBooks subscription cost?",
    "When is the final exam scheduled?",
    "Can I use AI tools on my homework?",
    "What happens if my program does not compile?",
    "How many late days do I get?",
    "Are lab absences penalized?",
    "What is the attendance policy for lab?",
    "Is the final exam cumulative?",
    "How are programming assignments graded?",
    "When is Exam 1?",
    "What is the academic integrity policy?",
]
SYLLABUS_OUT = [
    "What is the capital of France?",
    "Write me a Python script that scrapes Twitter.",
    "Who won the World Series in 2019?",
    "Give me a recipe for lasagna.",
    "Explain quantum entanglement.",
    "Should I buy Tesla stock?",
    "Tell me a joke about horses.",
    "Can I return these boots?",
    "How much is overnight shipping?",
    "What is your warranty on jackets?",
    "Help me write a cover letter.",
    "What time does the Super Bowl start?",
    "What are the symptoms of appendicitis?",
    "Who directed Inception?",
]
SUPPORT_IN = [
    "How long do I have to return an item?",
    "How much is expedited shipping?",
    "Can I cancel my order?",
    "What does the warranty cover?",
    "When will my refund post?",
    "Do you ship internationally?",
    "What payment methods do you accept?",
    "My package tracking has not updated in a week.",
    "Are final sale items returnable?",
    "Who pays for return shipping?",
    "How do I file a warranty claim?",
]
SUPPORT_OUT = [
    "What is the capital of France?",
    "Help me write a cover letter.",
    "Who is the president of Brazil?",
    "Explain the Pythagorean theorem.",
    "What time does the Super Bowl start?",
    "When is the final exam?",
    "What chapters does Exam 1 cover?",
    "Who is the course instructor?",
    "How do I bake sourdough?",
]


def build(path: Path) -> DocumentStore:
    store = DocumentStore(data_dir=Path(tempfile.mkdtemp()))
    store.add(path.name, path.read_bytes())
    return store


def score_all(store: DocumentStore, questions):
    out = []
    for q in questions:
        hits = store.index.search(q, top_k=6)
        out.append((relevance(hits[0] if hits else None), q))
    return out


def main() -> int:
    syllabus = build(ROOT / "examples" / "course_ta" / "cs1428_syllabus.md")
    support = build(ROOT / "examples" / "customer_care" / "returns_and_shipping.md")

    in_scope = score_all(syllabus, SYLLABUS_IN) + score_all(support, SUPPORT_IN)
    off_topic = score_all(syllabus, SYLLABUS_OUT) + score_all(support, SUPPORT_OUT)

    print("\nLOWEST-SCORING IN-SCOPE QUESTIONS (risk of false refusal)")
    for score, q in sorted(in_scope)[:8]:
        print("  {:.3f}  {}".format(score, q))

    print("\nHIGHEST-SCORING OFF-TOPIC QUESTIONS (risk of false answer)")
    for score, q in sorted(off_topic)[-8:]:
        print("  {:.3f}  {}".format(score, q))

    print("\nERRORS BY THRESHOLD  (n_in={}, n_out={})".format(
        len(in_scope), len(off_topic)))
    print("  {:>6}  {:>14}  {:>14}".format("thresh", "false refusals", "false answers"))
    for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        fn = sum(1 for s, _ in in_scope if s < t)
        fp = sum(1 for s, _ in off_topic if s >= t)
        print("  {:>6.2f}  {:>14}  {:>14}".format(t, fn, fp))

    print(
        "\nThe band where the two populations overlap is the uncertain zone.\n"
        "Questions landing there are sent to the LLM with a verification\n"
        "instruction rather than being auto-refused. See app/core/guard.py.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
