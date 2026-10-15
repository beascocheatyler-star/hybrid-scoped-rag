"""The scope guard — the part that makes this a *scoped* chatbot.

Two defences, layered, because either alone is insufficient:

1. **Retrieval gate (pre-LLM).** If the best match scores below the relevance
   floor, the question is off-topic by construction: nothing in scope speaks to
   it. We refuse without spending a token. This is the cheap, deterministic,
   un-jailbreakable layer — no prompt can talk its way past it, because the
   model is never called.

2. **Grounding instruction (in-LLM).** Retrieval can return weak-but-nonzero
   matches for a question the corpus doesn't actually answer (shared vocabulary,
   e.g. asking a refund-policy bot about "the return of the king"). So the
   system prompt also requires the model to answer *only* from the supplied
   context and to say so when the context falls short.

Layer 1 catches the obvious; layer 2 catches the subtle. A prompt-injection
attempt inside an uploaded document is addressed separately — see
``build_system_prompt``, which tells the model that document text is reference
material, never instructions.

**What changed from the predecessor.** The gate now reads a hybrid signal. The
lexical score remains primary and the dense score can only *rescue* a question,
never sink one — see ``relevance`` for why that asymmetry is deliberate.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.core.metadata import ScopeFilter
from app.core.retrieval import SearchHit, tokenize

# Greetings and meta-questions are in-scope conversationally even though they
# retrieve nothing. Refusing "hello" makes the bot feel broken.
_SMALL_TALK = re.compile(
    r"^\s*(hi|hey|hello|yo|good\s+(morning|afternoon|evening)|thanks?|thank\s+you|"
    r"ok(ay)?|got\s+it|bye|goodbye|see\s+ya)"
    # Allow a short trailing flourish: "hey there", "thanks so much".
    r"(\s+(there|all|again|you|so\s+much|a\s+lot))?\b[\s!.?]*$",
    re.IGNORECASE,
)

# Questions about the bot's own remit, which it should answer from its persona
# and document list rather than from retrieved passages.
_META_PATTERNS = re.compile(
    r"\b(what (can|do) you (do|help|answer)|who are you|what are you|"
    r"what documents|what do you know|what topics|how do you work|"
    r"what is your (scope|purpose))\b",
    re.IGNORECASE,
)


# The lexical gate scores a question as sqrt(bm25_normalized * coverage) — the
# geometric mean, which requires *both* signals to be respectable. Neither alone
# suffices:
#
#   * High BM25, low coverage  -> "What time does the Super Bowl start?" matches
#     a syllabus strongly on the single word "time".
#   * Low BM25, high coverage  -> "Help me write a cover letter" shares several
#     common words with a policy document without being about it.
#
# The geometric mean punishes a near-zero in either factor far harder than an
# average would, which is exactly the desired behaviour.
#
# Measured against the bundled corpora, in-scope and off-topic questions overlap
# in a narrow band (~0.40 to ~0.50): no single threshold is error-free there.
# Rather than pick a cutoff that is wrong in one direction, the band is treated
# as genuinely uncertain and handed to the LLM *with* the retrieved text and an
# explicit instruction to verify that it addresses the question. Lexical scoring
# decides the easy cases for free; the model adjudicates only the hard ones.
#
# See docs/scope-tuning.md for the measured distribution and how to re-tune.
RELEVANCE_FLOOR_DEFAULT = 0.40

# Above this, retrieval is confident enough to answer without hedging.
CONFIDENT_RELEVANCE = 0.50

# A dense match this strong promotes an otherwise-failing question into the
# borderline band.
#
# Calibrated by measurement, not by intuition — see scripts/measure_hybrid.py,
# which reproduces these numbers against the bundled corpora. With the default
# hashing embedder, best-hit dense similarity distributes as:
#
#     in-scope questions    min 0.115  median 0.195  max 0.361
#     off-topic questions   min 0.038  median 0.124  max 0.206
#
# The distributions overlap, so no threshold separates them cleanly. 0.25 sits
# just above the off-topic maximum: it admits the strongest paraphrase cases
# while refusing every off-topic question in the sample. Note that rescue only
# promotes to BORDERLINE, where the model is explicitly instructed to verify
# the context — so a false accept here costs one guarded LLM call, not a wrong
# answer.
#
# These values are specific to the hashing backend, whose cosine similarities
# are compressed into a narrow range because it measures character-n-gram
# overlap rather than meaning. A sentence-transformers backend spreads them
# much wider and wants a substantially higher floor (~0.5); re-run the script
# after switching backends. Set DENSE_RESCUE_FLOOR=1.0 to disable rescue.
DENSE_RESCUE_FLOOR = 0.25


class ScopeDecision:
    IN_SCOPE = "in_scope"
    # Retrieval is ambiguous: plausible but not convincing. Goes to the LLM with
    # a heightened instruction to refuse if the context does not actually answer.
    BORDERLINE = "borderline"
    OUT_OF_SCOPE = "out_of_scope"
    SMALL_TALK = "small_talk"
    META = "meta"
    NO_CORPUS = "no_corpus"
    # The metadata filter matched no documents at all. Distinct from
    # OUT_OF_SCOPE: the question may be perfectly answerable, just not within
    # the scope the user selected, so the remedy is to widen the filter rather
    # than to ask something else.
    EMPTY_SCOPE = "empty_scope"


@dataclass
class ScopeResult:
    decision: str
    hits: List[SearchHit]
    best_score: float
    reason: str
    coverage: float = 0.0
    dense_score: float = 0.0
    # Set when the dense retriever promoted a question the lexical gate alone
    # would have refused. Surfaced through the API because it is the single
    # clearest demonstration that hybrid retrieval is doing something BM25
    # could not.
    rescued_by_dense: bool = False
    scope_description: str = ""

    @property
    def allows_llm(self) -> bool:
        """Whether this decision should proceed to an LLM call."""
        return self.decision in {
            ScopeDecision.IN_SCOPE,
            ScopeDecision.BORDERLINE,
            ScopeDecision.SMALL_TALK,
            ScopeDecision.META,
        }

    @property
    def needs_verification(self) -> bool:
        """Whether the model should be told to double-check scope itself."""
        return self.decision == ScopeDecision.BORDERLINE


def lexical_relevance(hit: Optional[SearchHit]) -> float:
    """Calibrated lexical relevance in 0..1 for the best hit."""
    if hit is None:
        return 0.0
    return math.sqrt(max(0.0, hit.normalized) * max(0.0, hit.coverage))


def relevance(hit: Optional[SearchHit]) -> float:
    """Combined topical-relevance score in 0..1. Lexical remains primary.

    The dense signal is deliberately *not* averaged in. Two reasons:

    1. The gate's thresholds are calibrated against the lexical score's
       distribution. Blending a second signal with a different distribution
       would silently invalidate that calibration, and re-tuning it is a
       measurement exercise, not a guess (see docs/scope-tuning.md).
    2. The failure modes are asymmetric. A dense retriever returning a high
       similarity for an off-topic question is a *false accept* — the expensive
       error for a scoped bot, since it lets the system answer things it should
       refuse. A dense retriever missing an in-scope question is a false
       reject, which the lexical score usually catches anyway.

    So dense similarity enters only through ``classify`` as a one-way rescue:
    it can lift a question into the borderline band, where the model is
    explicitly instructed to verify scope before answering, but it can never
    push one over the confident threshold on its own.
    """
    return lexical_relevance(hit)


def classify(
    question: str,
    hits: List[SearchHit],
    relevance_floor: float,
    corpus_empty: bool,
    scope: Optional[ScopeFilter] = None,
    scope_matched_documents: Optional[int] = None,
    dense_rescue_floor: float = DENSE_RESCUE_FLOOR,
) -> ScopeResult:
    """Decide whether ``question`` may be answered from the scoped corpus."""
    best_hit = hits[0] if hits else None
    best = relevance(best_hit)
    coverage = best_hit.coverage if best_hit else 0.0
    best_dense = max((h.dense_score for h in hits), default=0.0)
    scope_description = scope.describe() if scope and not scope.is_empty else ""

    if corpus_empty:
        return ScopeResult(
            decision=ScopeDecision.NO_CORPUS,
            hits=[],
            best_score=0.0,
            reason="No documents have been uploaded yet.",
        )

    # A filter that matches nothing is a different failure from an off-topic
    # question, and conflating them would send the user to fix the wrong thing.
    if scope_matched_documents == 0:
        return ScopeResult(
            decision=ScopeDecision.EMPTY_SCOPE,
            hits=[],
            best_score=0.0,
            reason=(
                "The active scope ({}) matches no documents in the knowledge "
                "base.".format(scope_description or "the selected filter")
            ),
            scope_description=scope_description,
        )

    if _SMALL_TALK.match(question):
        return ScopeResult(
            decision=ScopeDecision.SMALL_TALK,
            hits=hits[:2],
            best_score=best,
            reason="Conversational greeting.",
            scope_description=scope_description,
        )

    if _META_PATTERNS.search(question):
        return ScopeResult(
            decision=ScopeDecision.META,
            hits=hits[:3],
            best_score=best,
            reason="Question about the assistant's own scope.",
            scope_description=scope_description,
        )

    # A question with no content words after stopword removal gives retrieval
    # nothing to work with; treat it as unanswerable rather than off-topic.
    if not tokenize(question):
        return ScopeResult(
            decision=ScopeDecision.OUT_OF_SCOPE,
            hits=[],
            best_score=0.0,
            reason="The question contains no searchable terms.",
            scope_description=scope_description,
        )

    if not hits or best < relevance_floor:
        # Hybrid rescue: the lexical gate would refuse, but the dense retriever
        # is confident. This is the paraphrase case — the question means what
        # the corpus says using words the corpus does not use.
        #
        # Rescue is suppressed once the operator raises the floor above the
        # confident-relevance band. Someone setting RELEVANCE_FLOOR=0.9 is
        # asking for strictly *more* conservative behaviour, and letting a
        # dense match quietly readmit questions would inverts that — raising
        # the floor would make the bot answer things a lower floor refused.
        # The floor must stay monotonic: higher floor, never more answers.
        rescue_allowed = relevance_floor <= CONFIDENT_RELEVANCE
        if hits and rescue_allowed and best_dense >= dense_rescue_floor:
            return ScopeResult(
                decision=ScopeDecision.BORDERLINE,
                hits=hits,
                best_score=best,
                coverage=coverage,
                dense_score=best_dense,
                rescued_by_dense=True,
                scope_description=scope_description,
                reason=(
                    "Lexical relevance {:.3f} is below the floor of {:.3f}, but "
                    "dense similarity {:.3f} meets the rescue threshold of "
                    "{:.3f} — the wording differs from the source. Asking the "
                    "model to verify the context before answering.".format(
                        best, relevance_floor, best_dense, dense_rescue_floor
                    )
                ),
            )

        return ScopeResult(
            decision=ScopeDecision.OUT_OF_SCOPE,
            hits=hits,
            best_score=best,
            coverage=coverage,
            dense_score=best_dense,
            scope_description=scope_description,
            reason=(
                "Relevance {:.3f} is below the floor of {:.3f} ({:.0%} of the "
                "question's key terms appear in scope, best dense similarity "
                "{:.3f}). Nothing here addresses this question.".format(
                    best, relevance_floor, coverage, best_dense
                )
            ),
        )

    if best < CONFIDENT_RELEVANCE:
        return ScopeResult(
            decision=ScopeDecision.BORDERLINE,
            hits=hits,
            best_score=best,
            coverage=coverage,
            dense_score=best_dense,
            scope_description=scope_description,
            reason=(
                "Relevance {:.3f} is in the uncertain band ({:.2f}–{:.2f}); "
                "asking the model to verify the context before answering.".format(
                    best, relevance_floor, CONFIDENT_RELEVANCE
                )
            ),
        )

    return ScopeResult(
        decision=ScopeDecision.IN_SCOPE,
        hits=hits,
        best_score=best,
        coverage=coverage,
        dense_score=best_dense,
        scope_description=scope_description,
        reason="Relevance {:.3f} with {:.0%} term coverage.".format(best, coverage),
    )


def refusal_message(
    result: ScopeResult,
    assistant_name: str,
    topic: str,
    document_names: Optional[List[str]] = None,
) -> str:
    """A deterministic, on-brand refusal. No LLM call required."""
    if result.decision == ScopeDecision.NO_CORPUS:
        return (
            "I don't have any documents yet, so there's nothing I can answer from. "
            "Upload the relevant files and ask me again."
        )

    if result.decision == ScopeDecision.EMPTY_SCOPE:
        return (
            "No documents match the scope you've selected ({}), so there's "
            "nothing for me to search. Widen the filter or clear it, and ask "
            "me again.".format(result.scope_description or "your filter")
        )

    lines = [
        "That falls outside what I can help with. I'm {}, and I only answer "
        "questions about {}.".format(assistant_name, topic)
    ]
    if result.scope_description:
        lines.append(
            "\nYou've also narrowed me to {}, which limits what I can see.".format(
                result.scope_description
            )
        )
    if document_names:
        shown = document_names[:5]
        suffix = (
            " (and {} more)".format(len(document_names) - len(shown))
            if len(document_names) > len(shown)
            else ""
        )
        lines.append(
            "\nMy knowledge base currently covers: {}{}.".format(
                ", ".join(shown), suffix
            )
        )
    lines.append("\nAsk me something in that area and I'll give you a sourced answer.")
    return "".join(lines)
