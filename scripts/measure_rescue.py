#!/usr/bin/env python3
"""Calibrate DENSE_RESCUE_FLOOR against the full labelled gate set.

measure_hybrid.py calibrates on 20 questions over the two bundled corpora.
This script sweeps the rescue floor over eval/datasets/gate.jsonl (160
questions, five corpora including three long public-domain texts) and prints
false-accept / false-refuse at each value plus the dense-score distribution.

    python scripts/measure_rescue.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.guard import RELEVANCE_FLOOR_DEFAULT, classify  # noqa: E402
from app.core.metadata import ScopeFilter  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval.run import build_notebooks, load  # noqa: E402

FLOORS = [0.20, 0.25, 0.30, 0.33, 0.36, 0.40, 0.45, 1.0]


def main() -> int:
    nbs = build_notebooks()
    pre = []
    for r in load("gate.jsonl"):
        st = nbs[r["notebook"]]
        sc = ScopeFilter.parse(r["scope"]) if r.get("scope") else None
        pre.append((r, st.index.search(r["question"], top_k=6, scope=sc), sc, st))

    dense = {"answer": [], "refuse": []}
    for r, h, _sc, _st in pre:
        dense[r["expected"]].append(max((x.dense_score for x in h), default=0.0))
    print("Gate set: {} answerable, {} off-topic".format(
        len(dense["answer"]), len(dense["refuse"])))
    for k, label in (("answer", "in-scope "), ("refuse", "off-topic")):
        d = dense[k]
        print("{} dense: min {:.3f}  median {:.3f}  max {:.3f}".format(
            label, min(d), statistics.median(d), max(d)))

    print("\n{:>12} {:>9} {:>9} {:>17} {:>17}".format(
        "rescue_floor", "FA", "FR", "rescued off-topic", "rescued in-scope"))
    for rf in FLOORS:
        exp, ans, r_off, r_in = [], [], 0, 0
        for r, h, sc, st in pre:
            c = classify(r["question"], h, RELEVANCE_FLOOR_DEFAULT, st.is_empty,
                         sc, st.count_matching(sc), rf)
            exp.append(r["expected"])
            ans.append(c.allows_llm)
            if c.rescued_by_dense:
                if r["expected"] == "refuse":
                    r_off += 1
                else:
                    r_in += 1
        fa, fr = M.gate_rates(exp, ans)
        print("{:>12.2f} {:>9.3f} {:>9.3f} {:>17} {:>17}".format(rf, fa, fr, r_off, r_in))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
