"""Unit tests for eval/metrics.py and the baseline regression gate."""

from __future__ import annotations

import pytest

from eval import metrics as M
from eval.run import check_baseline


def test_span_in_ignores_case_and_whitespace():
    assert M.span_in("The Final  exam\nis Monday", ["final exam is"])
    assert not M.span_in("nothing here", ["final"])


def test_recall_and_mrr():
    ranks = [1, 3, None, 7]
    assert M.recall_at_k(ranks, 6) == 0.5
    assert M.mrr(ranks) == pytest.approx((1 + 1 / 3 + 0 + 1 / 7) / 4)


def test_ndcg_perfect_and_zero():
    assert M.ndcg_at_k([[True, False]]) == 1.0
    assert M.ndcg_at_k([[False, False]]) == 0.0
    assert 0 < M.ndcg_at_k([[False, True]]) < 1


def test_gate_rates():
    fa, fr = M.gate_rates(
        ["answer", "answer", "refuse", "refuse"], [True, False, True, False]
    )
    assert fa == 0.5 and fr == 0.5


def test_percentile():
    assert M.percentile([5, 1, 3, 2, 4], 50) == 3
    assert M.percentile([], 95) == 0.0


def test_baseline_gate_respects_direction_and_tolerance():
    base = {"retrieval_recall@6": 0.90, "gate_false_accept": 0.10,
            "injection_resistance": 1.0}
    assert check_baseline(dict(base), base) == []
    assert check_baseline({**base, "retrieval_recall@6": 0.89}, base) == []
    assert check_baseline({**base, "retrieval_recall@6": 0.85}, base)
    assert check_baseline({**base, "gate_false_accept": 0.15}, base)
    assert check_baseline({**base, "injection_resistance": 0.9}, base)


def test_offline_eval_matches_committed_baseline():
    """The same check CI runs: no metric regresses >2 points."""
    import json
    from pathlib import Path

    from eval.run import BASELINE, UNGATED, run

    res = run(offline=True)
    gated = {k: v for k, v in res.items() if not k.startswith("_") and k not in UNGATED}
    assert check_baseline(gated, json.loads(Path(BASELINE).read_text())) == []
