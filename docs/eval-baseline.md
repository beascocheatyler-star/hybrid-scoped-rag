# Eval baseline (M0)

Recorded 2026-11-23 on the M0 branch, default configuration
(`EMBEDDING_BACKEND=hashing`, `RELEVANCE_FLOOR=0.40`, `DENSE_RESCUE_FLOOR=0.36`,
`TOP_K=6`). Reproduce with:

```bash
python -m eval.run --offline
```

Machine-readable copy: `eval/baseline.json` (CI compares against it).

## Corpora

| Notebook | Source | Words |
|---|---|---|
| syllabus | `examples/course_ta/cs1428_syllabus.md` (bundled) | ~1.1k |
| support | `examples/customer_care/returns_and_shipping.md` (bundled) | ~0.9k |
| both | the two above, tagged `course=cs1428` / `dept=support` | — |
| constitution | US Constitution, Articles I–VII (government document) | ~4.5k |
| faraday | Faraday, *Chemical History of a Candle*, Lecture I, 1861 (lecture transcript) | ~5.7k |
| darwin | Darwin, *Origin of Species* (1859), ch. III (science text chapter) | ~6.2k |

The three new texts are public domain, retrieved from Wikisource, and stored
under `examples/public_domain/`. `tests/test_eval_datasets.py` checks that
every labelled span and reference fact really occurs in its corpus.

## Results

| Metric | Value | n |
|---|---|---|
| retrieval recall@6 | 0.840 | 125 |
| retrieval MRR | 0.711 | 125 |
| retrieval nDCG@10 | 0.748 | 125 |
| recall@6, lexical | 1.000 | 80 |
| recall@6, paraphrase (no shared wording) | 0.700 | 30 |
| recall@6, non-English | 0.267 | 15 |
| gate false-accept (reaches the LLM, incl. borderline) | 0.188 | 80 off-topic |
| gate false-accept (ruled `in_scope` outright) | 0.025 | 80 off-topic |
| gate false-refuse | 0.075 | 80 answerable |
| follow-ups passing the gate | 0.467 | 15 |
| context fact recall | 1.000 | 23 facts |
| citation precision / recall | n/a (M4) | — |
| faithfulness | n/a (live only) | — |
| injection resistance (structural) | 1.000 | 12 |
| search latency p50 / p95 | 0.2 / 1.1 ms | 125 |
| prompt tokens per answer (approx.) | ~2310 | 16 |

## What the numbers say

- **Lexical retrieval is saturated (1.000).** The gaps are paraphrase (0.70)
  and non-English (0.27). The hashing embedder's character n-grams catch some
  Romance-language cognates (*examen final*); CJK and Cyrillic get nothing.
- **False refusals (6/80):** three non-English questions (ASCII tokenizer,
  known defect #4), one paraphrase with no shared words ("Can I send back shoes
  I already wore?"), "What room is the lab in?" (0.277) and "How slowly does
  the elephant breed?" (0.324: the corpus says "breeder", and the tokenizer
  does no stemming).
- **Confident false accepts (2/80):** "Tell me about the return of the king"
  (0.518) and "What is a gift of the Magi?" (0.523) against the support policy.
  These are the deliberate word-overlap traps.
- **Borderline false accepts (13/80):** go to the model with a verify-scope
  instruction. Most are single-word overlaps ("final", "capital", "court").
- **Follow-ups (7/15)** confirm known defect #5. M3 targets at least 80%.

## Calibration change made in M0

On the full gate set the old dense-rescue floor (0.25) rescued 5 off-topic
questions and 0 answerable ones. It is now 0.36, which cuts false accepts from
0.250 to 0.188 with false refusals unchanged. The sweep is reproducible with
`python scripts/measure_rescue.py`, and the full table is in
`docs/scope-tuning.md`.

## Injection set

Each of the 6 injected sources has an in-scope probe and an off-topic probe.
Offline, a case passes if no model call happens (nothing could obey the
injection), or if the call keeps the injected text out of the system prompt
with the untrusted-data rule present. One in-scope probe ("How long can I
borrow books?" against "may be borrowed") is refused by the gate because of
missing stemming. That is a false refusal, the safe direction, and it is
reported separately rather than counted as resistance.

## Targets set by the roadmap

- M3: recall@6 +15 points with a semantic embedder, no regression with
  hashing, false accepts must not rise, false refusals should fall, follow-ups
  at least 0.80, non-Latin questions answered.
- M4: citation precision at least 0.85, citation recall at least 0.75.
