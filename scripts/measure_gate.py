#!/usr/bin/env python3
"""Calibrate the scope gate against the labelled eval gate set.

Sweeps RELEVANCE_FLOOR and DENSE_RESCUE_FLOOR over eval/datasets/gate.jsonl
(80 answerable, 80 off-topic questions across six notebooks) for the
configured embedder, prints the error table, and recommends floors:

* dense rescue floor: the larger of (a) the smallest swept value that
  rescues no off-topic question and (b) the off-topic dense maximum rounded
  up, so the floor keeps a margin above every off-topic question seen;
* relevance floor: among floors whose false-accept rate does not exceed the
  committed baseline, the one with the fewest false refusals (ties: higher,
  i.e. safer).

    python scripts/measure_gate.py                 # print only
    python scripts/measure_gate.py --write [--db data/hsr.sqlite3]

--write stores a row in the ``calibrations`` table for this embedder
signature; the app then uses those floors unless RELEVANCE_FLOOR /
DENSE_RESCUE_FLOOR are set explicitly in the environment.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.guard import classify  # noqa: E402
from app.core.metadata import ScopeFilter  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval.run import BASELINE, build_notebooks, load  # noqa: E402

RELEVANCE = [0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]
RESCUE = [0.20, 0.25, 0.30, 0.33, 0.36, 0.40, 0.45, 0.50, 0.60, 0.70, 1.0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--db", default=str(ROOT / "data" / "hsr.sqlite3"))
    a = ap.parse_args(argv)

    nbs = build_notebooks()
    sig = next(iter(nbs.values()))._embedding_signature
    pre = []
    for r in load("gate.jsonl"):
        st = nbs[r["notebook"]]
        sc = ScopeFilter.parse(r["scope"]) if r.get("scope") else None
        pre.append((r, st.index.search(r["question"], top_k=6, scope=sc), sc, st))

    def run(rel, res):
        exp, ans, conf, r_off, r_in = [], [], [], 0, 0
        for r, h, sc, st in pre:
            c = classify(r["question"], h, rel, st.is_empty, sc, st.count_matching(sc), res)
            exp.append(r["expected"])
            ans.append(c.allows_llm)
            conf.append(c.decision == "in_scope")
            if c.rescued_by_dense:
                r_off += r["expected"] == "refuse"
                r_in += r["expected"] == "answer"
        fa, fr = M.gate_rates(exp, ans)
        fc, _ = M.gate_rates(exp, conf)
        return fa, fc, fr, r_off, r_in

    dense = {"answer": [], "refuse": []}
    for r, h, _, _ in pre:
        dense[r["expected"]].append(max((x.dense_score for x in h), default=0.0))
    print("Embedder {}  |  {} answerable, {} off-topic".format(sig, len(dense["answer"]), len(dense["refuse"])))
    for k, lbl in (("answer", "in-scope "), ("refuse", "off-topic")):
        d = dense[k]
        print("  {} best dense: min {:.3f} median {:.3f} max {:.3f}".format(lbl, min(d), statistics.median(d), max(d)))

    print("\nDense rescue sweep (relevance floor 0.42)")
    print("  {:>7} {:>7} {:>7} {:>7} {:>13} {:>12}".format("rescue", "FA", "FAconf", "FR", "rescued off", "rescued in"))
    rescue_pick = 1.0
    for res in RESCUE:
        fa, fc, fr, ro, ri = run(0.42, res)
        print("  {:>7.2f} {:>7.3f} {:>7.3f} {:>7.3f} {:>13} {:>12}".format(res, fa, fc, fr, ro, ri))
        if ro == 0 and rescue_pick == 1.0:
            rescue_pick = res

    # Margin: never below the off-topic dense maximum (the documented rule),
    # even if this particular sample happens to need less.
    import math
    rescue_pick = max(rescue_pick, math.ceil(max(dense["refuse"]) * 100) / 100)

    base = json.loads(Path(BASELINE).read_text()) if Path(BASELINE).exists() else {}
    fa_cap = base.get("gate_false_accept", 1.0)
    print("\nRelevance floor sweep (rescue {:.2f}); baseline false-accept cap {:.3f}".format(rescue_pick, fa_cap))
    print("  {:>7} {:>7} {:>7} {:>7}".format("floor", "FA", "FAconf", "FR"))
    best = None
    for rel in RELEVANCE:
        fa, fc, fr, _, _ = run(rel, rescue_pick)
        print("  {:>7.2f} {:>7.3f} {:>7.3f} {:>7.3f}".format(rel, fa, fc, fr))
        if fa <= fa_cap + 1e-9 and (best is None or fr < best[1] or (fr == best[1] and rel > best[0])):
            best = (rel, fr)
    rel_pick = best[0] if best else RELEVANCE[-1]
    print("\nRecommended: RELEVANCE_FLOOR={:.2f}  DENSE_RESCUE_FLOOR={:.2f}".format(rel_pick, rescue_pick))

    if a.write:
        from app.core.db import Database, utcnow

        db = Database(Path(a.db))
        db.execute(
            "INSERT INTO calibrations(embedder_signature, relevance_floor, dense_rescue_floor, "
            "measured_at, report_path) VALUES (?,?,?,?, 'scripts/measure_gate.py') "
            "ON CONFLICT(embedder_signature) DO UPDATE SET relevance_floor = excluded.relevance_floor, "
            "dense_rescue_floor = excluded.dense_rescue_floor, measured_at = excluded.measured_at",
            (sig, rel_pick, rescue_pick, utcnow()))
        print("Wrote calibration for {} to {}".format(sig, a.db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
