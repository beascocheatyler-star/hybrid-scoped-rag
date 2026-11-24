"""Pure metric functions for the offline/live eval. No I/O, no app imports."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def span_in(text: str, spans: Sequence[str]) -> bool:
    """True if any labelled span occurs in ``text`` (case/whitespace-insensitive).

    Labels are spans, not chunk ids, so the dataset survives re-chunking.
    """
    t = _norm(text)
    return any(_norm(s) in t for s in spans)


def first_relevant_rank(texts: Sequence[str], spans: Sequence[str]) -> Optional[int]:
    for i, t in enumerate(texts, start=1):
        if span_in(t, spans):
            return i
    return None


def recall_at_k(ranks: Sequence[Optional[int]], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: Sequence[Optional[int]]) -> float:
    if not ranks:
        return 0.0
    return sum((1.0 / r) if r else 0.0 for r in ranks) / len(ranks)


def ndcg_at_k(rel_lists: Sequence[Sequence[bool]], k: int = 10) -> float:
    """Binary-relevance nDCG@k averaged over queries."""
    if not rel_lists:
        return 0.0
    total = 0.0
    for rels in rel_lists:
        rels = list(rels)[:k]
        dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rels) if r)
        n_rel = sum(1 for r in rels if r)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
        total += (dcg / idcg) if idcg else 0.0
    return total / len(rel_lists)


def gate_rates(expected: Sequence[str], answered: Sequence[bool]):
    """Return (false_accept_rate, false_refuse_rate).

    false accept  = off-topic question that was let through to the LLM
    false refuse  = answerable question that was refused
    """
    refuse = [a for e, a in zip(expected, answered) if e == "refuse"]
    answer = [a for e, a in zip(expected, answered) if e == "answer"]
    fa = (sum(1 for a in refuse if a) / len(refuse)) if refuse else 0.0
    fr = (sum(1 for a in answer if not a) / len(answer)) if answer else 0.0
    return fa, fr


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(math.ceil(p / 100.0 * len(s))) - 1))
    return s[idx]
