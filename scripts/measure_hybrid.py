#!/usr/bin/env python3
"""Measure what hybrid retrieval and scoping actually buy.

Runs entirely offline — no API key, no network, no cost. Everything the README
claims about the dense rescue floor and the lexical/dense split is reproduced
here, so a reviewer can check the numbers rather than trust them.

    python scripts/measure_hybrid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tempfile  # noqa: E402

from app.core.embedding import HashingEmbedder  # noqa: E402
from app.core.guard import (  # noqa: E402
    CONFIDENT_RELEVANCE,
    DENSE_RESCUE_FLOOR,
    RELEVANCE_FLOOR_DEFAULT,
    classify,
)
from app.core.metadata import ScopeFilter  # noqa: E402
from app.core.store import DocumentStore  # noqa: E402

EXAMPLES = REPO_ROOT / "examples"

# Questions the bundled corpora genuinely answer.
IN_SCOPE = [
    "When is the final exam?",
    "What is the late work policy?",
    "How is the final grade calculated?",
    "When are office hours?",
    "What happens if I miss an exam?",
    "How many programming assignments are there?",
    "How do I return a pair of boots?",
    "How long does shipping take?",
    "Can I get a refund on a damaged item?",
    "What is the exchange policy?",
]

# Questions nothing in the corpora addresses.
OUT_OF_SCOPE = [
    "Who won the Super Bowl in 1998?",
    "Write me a poem about the ocean",
    "What is the capital of France?",
    "Explain quantum entanglement",
    "What is the best pizza recipe?",
    "How do I change a flat tyre?",
    "Translate 'hello' into Japanese",
    "Which stocks should I buy this year?",
    "What is the weather tomorrow?",
    "Tell me a joke about cats",
]

# In-scope questions deliberately phrased with words the documents avoid.
# These are what the dense retriever is supposed to help with.
PARAPHRASES = [
    "when do I sit the midterm",
    "what is the enrolment deadline",
    "syllabi requirements",
    "whats the returns window",
    "can I send back footwear",
]


def build_store(tmp):
    store = DocumentStore(Path(tmp), embedder=HashingEmbedder())
    store.add(
        "cs1428_syllabus.md",
        (EXAMPLES / "course_ta" / "cs1428_syllabus.md").read_bytes(),
        metadata={"course": "CS1428", "term": "Fall2025"},
    )
    store.add(
        "returns_and_shipping.md",
        (EXAMPLES / "customer_care" / "returns_and_shipping.md").read_bytes(),
        metadata={"dept": "support"},
    )
    return store


def summarize(values):
    if not values:
        return "n/a"
    ordered = sorted(values)
    return "min {:.3f}  median {:.3f}  max {:.3f}".format(
        ordered[0], ordered[len(ordered) // 2], ordered[-1]
    )


def rule(title):
    print("\n{}\n{}".format(title, "-" * len(title)))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        store = build_store(tmp)

        print("Corpus: {} documents, {} passages".format(
            len(store.list_documents()), store.total_chunks
        ))
        print("Embedder: {} ({}d)".format(
            store.embedder.name, store.embedder.dimensions
        ))

        # ------------------------------------------------------ score split
        rule("1. Score distributions")

        def scores(questions):
            lex, dense = [], []
            for q in questions:
                hits = store.index.search(q, top_k=6)
                result = classify(q, hits, RELEVANCE_FLOOR_DEFAULT, store.is_empty)
                lex.append(result.best_score)
                dense.append(max((h.dense_score for h in hits), default=0.0))
            return lex, dense

        in_lex, in_dense = scores(IN_SCOPE)
        out_lex, out_dense = scores(OUT_OF_SCOPE)

        print("in-scope   lexical:  {}".format(summarize(in_lex)))
        print("off-topic  lexical:  {}".format(summarize(out_lex)))
        print("in-scope   dense:    {}".format(summarize(in_dense)))
        print("off-topic  dense:    {}".format(summarize(out_dense)))
        print()
        print("Floor {:.2f} / confident {:.2f} / dense rescue {:.2f}".format(
            RELEVANCE_FLOOR_DEFAULT, CONFIDENT_RELEVANCE, DENSE_RESCUE_FLOOR
        ))
        print(
            "The rescue floor must exceed the off-topic dense maximum "
            "({:.3f}) or it would readmit off-topic questions.".format(
                max(out_dense) if out_dense else 0.0
            )
        )

        # ------------------------------------------------------ gate outcomes
        rule("2. Gate decisions")
        correct = sum(
            1
            for q in IN_SCOPE
            if classify(
                q, store.index.search(q, top_k=6), RELEVANCE_FLOOR_DEFAULT, False
            ).allows_llm
        )
        refused = sum(
            1
            for q in OUT_OF_SCOPE
            if not classify(
                q, store.index.search(q, top_k=6), RELEVANCE_FLOOR_DEFAULT, False
            ).allows_llm
        )
        print("in-scope admitted:  {}/{}".format(correct, len(IN_SCOPE)))
        print("off-topic refused:  {}/{}".format(refused, len(OUT_OF_SCOPE)))

        # ------------------------------------------------- retriever overlap
        rule("3. What each retriever contributes")
        both = lexical_only = dense_only = 0
        for q in IN_SCOPE + PARAPHRASES:
            for hit in store.index.search(q, top_k=6):
                if hit.lexical_rank and hit.dense_rank:
                    both += 1
                elif hit.lexical_rank:
                    lexical_only += 1
                else:
                    dense_only += 1
        total = both + lexical_only + dense_only
        print("passages found by both:          {:3d}  ({:.0%})".format(
            both, both / total if total else 0
        ))
        print("passages found only by lexical:  {:3d}  ({:.0%})".format(
            lexical_only, lexical_only / total if total else 0
        ))
        print("passages found only by dense:    {:3d}  ({:.0%})".format(
            dense_only, dense_only / total if total else 0
        ))
        print(
            "\nPassages found only by dense are ones pure BM25 would have "
            "missed entirely."
        )

        # ------------------------------------------------------- scope leak
        rule("4. Scoping is applied before scoring")
        question = "How do I return a pair of boots?"
        unscoped = store.index.search(question, top_k=3)
        scoped = store.index.search(
            question, top_k=3, scope=ScopeFilter.parse({"course": "CS1428"})
        )

        print("question: {!r}".format(question))
        print("unscoped -> {} (coverage {:.2f})".format(
            [h.chunk.doc_name for h in unscoped],
            unscoped[0].coverage if unscoped else 0.0,
        ))
        print("scoped to course=CS1428 -> {} (coverage {:.2f})".format(
            [h.chunk.doc_name for h in scoped],
            scoped[0].coverage if scoped else 0.0,
        ))
        print(
            "\nThe filter is applied before scoring, so the returns policy is\n"
            "not merely demoted — it is never a candidate. Coverage is also\n"
            "computed over the scoped subset rather than the whole corpus, so a\n"
            "question cannot pass the gate on terms that exist only in\n"
            "documents the filter excluded. (With this two-document corpus both\n"
            "coverage figures happen to coincide; see\n"
            "tests/test_hybrid_retrieval.py::"
            "test_coverage_is_computed_over_the_scoped_subset for a case that\n"
            "isolates the difference.)"
        )

        decision = classify(
            question,
            scoped,
            RELEVANCE_FLOOR_DEFAULT,
            store.is_empty,
            scope=ScopeFilter.parse({"course": "CS1428"}),
            scope_matched_documents=1,
        )
        print("scoped decision: {}".format(decision.decision))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
