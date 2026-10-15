# Tuning the scope gate

How the relevance threshold was chosen, what the measured numbers are, and how
to re-tune for your own documents.

Reproduce everything here with:

```bash
python scripts/measure_scope.py
```

---

## The score

```
relevance = √(bm25_normalized × term_coverage)
```

| Component | Meaning | Range |
|---|---|---|
| `bm25_normalized` | Best passage's BM25 score, divided by the maximum that query could achieve | 0–1 |
| `term_coverage` | Fraction of the question's content words occurring anywhere in the corpus | 0–1 |

The geometric mean requires **both** signals to be respectable. An arithmetic
mean would let a single strong signal mask a near-zero one, which is the exact
failure this is designed to prevent.

---

## Measured distribution

Against the two bundled corpora, with 26 in-scope and 23 off-topic questions.

### Lowest-scoring in-scope questions (risk of false refusal)

| Score | Question |
|---|---|
| 0.440 | Are lab absences penalized? |
| 0.446 | How much does the zyBooks subscription cost? |
| 0.511 | What happens if I miss an exam? |
| 0.515 | When will my refund post? |
| 0.519 | My package tracking has not updated in a week. |

### Highest-scoring off-topic questions (risk of false answer)

| Score | Question | Why it scores at all |
|---|---|---|
| 0.338 | Write me a Python script that scrapes Twitter. | "write", "program" |
| 0.385 | What time does the Super Bowl start? | "time" |
| 0.446 | Can I return these boots? | "return" |
| 0.451 | When is the final exam? *(asked of the support bot)* | "final" |
| 0.499 | Help me write a cover letter. | "help", "write", "cover" |

### Errors by threshold

| Threshold | False refusals (of 26) | False answers (of 23) |
|---|---|---|
| 0.30 | 0 | 9 |
| 0.35 | 0 | 7 |
| **0.40** | **0** | **4** |
| 0.45 | 2 | 3 |
| 0.50 | 2 | 0 |
| 0.55 | 5 | 0 |
| 0.60 | 11 | 0 |

---

## Why 0.40, and why a band

The two populations **overlap between roughly 0.44 and 0.50**. No single
threshold separates them cleanly:

- Pick 0.40 → 4 off-topic questions get through.
- Pick 0.50 → 2 legitimate questions get refused.

Rather than accept either, the design splits the difference by treating the
overlap as what it actually is — uncertainty:

| Range | Decision | Behaviour |
|---|---|---|
| `< 0.40` | `out_of_scope` | Refused immediately. No LLM call, no cost. |
| `0.40 – 0.50` | `borderline` | Sent to the LLM **with the retrieved text** and an instruction to verify it genuinely answers the question before responding. |
| `≥ 0.50` | `in_scope` | Answered normally. |

The threshold is set where **false refusals are zero**. That asymmetry is
intentional: a wrongly refused question is a visibly broken product, while a
borderline question merely costs one API call to a layer that can read the
passages and judge properly. Cheap lexical scoring handles the easy 90%; the
model adjudicates only the genuinely ambiguous cases.

---

## Re-tuning for your own documents

`RELEVANCE_FLOOR` in `.env` controls the lower bound.

| Symptom | Fix |
|---|---|
| Bot refuses legitimate questions | **Lower** it (try 0.30) |
| Bot answers off-topic questions | **Raise** it (try 0.50) |

To measure properly rather than guess, edit the question lists at the top of
`scripts/measure_scope.py` to reflect your corpus and your users' real questions,
then run it. Pick the threshold where false refusals reach zero.

Corpus-size note: the scores are stable across corpus sizes because coverage is
presence-based. An earlier version weighted terms by how many chunks contained
them, which inverted badly on small corpora — on a 7-chunk syllabus it rated
"exam" as uninformative (0.29) and "boots" as highly informative (1.00), exactly
backwards. In a single-subject knowledge base, the most widespread term *is* the
subject. That approach was removed; see the comment in `retrieval.py::_coverage`.

---

## Other levers

| Setting | Effect on scope behaviour |
|---|---|
| `TOP_K` | More passages give the model more to work with, but dilute precision. Only the *best* hit drives the gate, so this does not change the decision — only the answer quality. |
| `CHUNK_SIZE` | Smaller chunks sharpen BM25 (less dilution per passage) but fragment context. |
| `CHUNK_OVERLAP` | Higher overlap helps when answers straddle a chunk boundary. |
| `STRICT_SCOPE` | Adds an explicit instruction refusing unrelated code/essay/translation requests even when retrieval passes. |

---

## The dense rescue

A question can fail the lexical gate and still be legitimate — it just used
different words than the documents do. The dense retriever gets one chance to
say so: if best-hit cosine similarity reaches `DENSE_RESCUE_FLOOR`, the
question is promoted to `borderline` instead of being refused.

Promotion is to `borderline`, never to `in_scope`. The model then receives the
strictest of the three instructions, explicitly told the passages were matched
by meaning rather than wording and may be spurious. A false accept therefore
costs one *guarded* LLM call, not a wrong answer.

### Why the signals are not averaged

The gate reads lexical relevance alone, with dense similarity acting only as a
one-way override. Two reasons:

1. The thresholds above are calibrated against the lexical score's
   distribution. Blending in a second signal with a different distribution
   would silently invalidate that calibration.
2. The failure modes are asymmetric. A dense retriever scoring an off-topic
   question highly is a **false accept** — the expensive error for a scoped
   bot. A dense retriever missing an in-scope question is a false reject, which
   the lexical score usually catches anyway.

### Rescue is suppressed at high floors

If `RELEVANCE_FLOOR` is raised above `CONFIDENT_RELEVANCE` (0.50), rescue is
disabled. Someone setting `RELEVANCE_FLOOR=0.9` is asking for strictly more
conservative behaviour; letting a dense match quietly readmit questions would
invert that, so raising the floor could make the bot answer *more*. The floor
must stay monotonic.

### Calibrating the floor

Run `python scripts/measure_hybrid.py`. With the default hashing embedder:

```
in-scope questions    min 0.115  median 0.195  max 0.361
off-topic questions   min 0.038  median 0.124  max 0.206
```

The distributions overlap, so no threshold separates them cleanly. The default
`0.25` sits just above the off-topic maximum.

**Re-run this after changing `EMBEDDING_BACKEND`.** The hashing embedder
compresses similarities into a narrow band because it measures character-n-gram
overlap; a sentence-transformers backend spreads them much wider and wants a
substantially higher floor (~0.5). A floor tuned for one backend is meaningless
for the other.

---

## The remaining limitation

Hybrid retrieval narrows the vocabulary gap without closing it.

"What textbook do I need?" against a syllabus headed *Required materials* is
still refused. The dense retriever now surfaces candidate passages — BM25
returned literally nothing — but the default hashing embedder scores character
n-grams, not meaning, and "textbook" shares few n-grams with "required
materials". Its similarity lands near 0.02, far under the rescue floor.

This is asserted by `test_vocabulary_gap_is_narrowed_but_not_closed`, which
checks both halves: that retrieval is no longer empty-handed, *and* that the
gate still refuses. The behaviour is recorded rather than discovered later.

Closing it properly needs true semantic embeddings:

```bash
pip install sentence-transformers
export EMBEDDING_BACKEND=sentence-transformers
python scripts/measure_hybrid.py   # then re-tune DENSE_RESCUE_FLOOR
```

The cost is a ~90 MB model download and slower ingest. Since the retrieval
score runs on *every* question to power the gate, that cost is paid constantly
— which is why the zero-dependency hashing backend remains the default.

A practical mitigation that needs no code change: include a glossary or FAQ in
the uploaded corpus. It supplies the vocabulary users actually ask in.
