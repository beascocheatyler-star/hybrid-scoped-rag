#!/usr/bin/env python3
"""Calibrate SUPPORT_FLOOR (citation verification, roadmap C3).

For every labelled retrieval question, take the sentence containing the
relevant span and score it against
  * the chunk that contains it            -> should count as supported
  * a random other chunk, same notebook   -> should not
and paraphrased variants (content words dropped) to mimic a model restating
rather than quoting. Prints both distributions and the error rates by floor.

    python scripts/measure_support.py
"""

from __future__ import annotations

import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.citations import split_sentences, support_score  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval.run import build_notebooks, load  # noqa: E402

FLOORS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def restate(sentence: str, rng: random.Random) -> str:
    """Drop ~30% of words: a crude paraphrase, so support is not only
    measured on verbatim quotes."""
    words = sentence.split()
    keep = [w for w in words if rng.random() > 0.3] or words
    return " ".join(keep)


def main() -> int:
    rng = random.Random(5)
    nbs = build_notebooks()
    pos, neg = [], []
    for r in load("retrieval.jsonl"):
        if r["kind"] != "lexical":
            continue
        st = nbs[r["notebook"]]
        chunks = st.index._chunks
        rel = next((c for c in chunks if M.span_in(c.text, r["relevant_spans"])), None)
        if rel is None:
            continue
        sent = next((s for s in split_sentences(rel.text)
                     if M.span_in(s, r["relevant_spans"])), None)
        if not sent or len(sent.split()) < 4:
            continue
        others = [c for c in chunks if c.chunk_id != rel.chunk_id
                  and not M.span_in(c.text, r["relevant_spans"])]
        for variant in (sent, restate(sent, rng)):
            pos.append(support_score(variant, rel.text, st.embedder))
            if others:
                neg.append(support_score(variant, rng.choice(others).text, st.embedder))
    print("pairs: {} supported, {} unsupported".format(len(pos), len(neg)))
    for name, xs in (("supported  ", pos), ("unsupported", neg)):
        print("  {} min {:.3f} p10 {:.3f} median {:.3f} max {:.3f}".format(
            name, min(xs), sorted(xs)[len(xs) // 10], statistics.median(xs), max(xs)))
    print("\n  floor  false-weak  false-supported")
    for f in FLOORS:
        fw = sum(1 for x in pos if x < f) / len(pos)
        fs = sum(1 for x in neg if x >= f) / len(neg)
        print("  {:.2f}   {:9.3f}   {:14.3f}".format(f, fw, fs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
