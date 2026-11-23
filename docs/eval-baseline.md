# Eval baseline (M0)

Recorded 2026-11-23 at the M0 branch, default configuration
(`EMBEDDING_BACKEND=hashing`, `RELEVANCE_FLOOR=0.40`, `DENSE_RESCUE_FLOOR=0.25`,
`TOP_K=6`). Reproduce with:

```bash
python -m eval.run --offline
```

Machine-readable copy: `eval/baseline.json` (CI compares against it).

| Metric | Value | n |
|---|---|---|
| retrieval recall@6 | 0.922 | 77 |
| retrieval MRR | 0.758 | 77 |
| retrieval nDCG@10 | 0.804 | 77 |
| recall@6, lexical questions | 1.000 | 56 |
| recall@6, paraphrases | 0.800 | 15 |
| recall@6, non-English | 0.500 | 6 |
| gate false-accept (reaches LLM) | 0.250 | 40 off-topic |
| gate false-accept (confident `in_scope`) | 0.050 | 40 off-topic |
| gate false-refuse | 0.125 | 40 answerable |
| follow-ups passing the gate | 0.467 | 15 |
| context fact recall | 1.000 | 18 facts |
| citation precision / recall | n/a (M4) | — |
| faithfulness | n/a (live only) | — |
| injection resistance (structural) | 1.000 | 6 |
| search latency p50 / p95 | 0.2 / 0.3 ms | 77 |
| prompt tokens per answer (approx.) | ~2285 | 10 |

## What the numbers say

- **Lexical retrieval is saturated** on these corpora (1.000). The gaps are
  paraphrase (0.80) and non-English (0.50; the hashing embedder's character
  n-grams catch cognates like *examen final*, but CJK and Cyrillic get
  nothing).
- **Gate false refusals (5/40):** three non-English questions (ASCII
  tokenizer: known defect #4), one paraphrase with no shared words ("Can I send
  back shoes I already wore?"), and "What room is the lab in?" (relevance
  0.277, the words appear but not together).
- **Confident false accepts (2/40):** "Tell me about the return of the king"
  (0.518) and "What is a gift of the Magi?" (0.523) against the support policy.
  Both are the adversarial lexical-overlap cases this set is meant to contain.
  The eight other false accepts land in `borderline` and go to the model with a
  verify-scope instruction.
- **Follow-ups (7/15)** confirm known defect #5. M3 targets ≥ 80%.

## Targets set by the roadmap

- M3: recall@6 +15 points with a semantic embedder; no regression with
  hashing; false-accept must not rise; false-refuse should fall; follow-ups
  ≥ 0.80; non-Latin questions answered.
- M4: citation precision ≥ 0.85, citation recall ≥ 0.75.
