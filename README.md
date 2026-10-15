# Hybrid Scoped RAG

A retrieval-augmented chatbot with two properties most RAG demos skip:

1. **Hybrid retrieval.** BM25 and dense vectors run in parallel and their
   rankings are fused with Reciprocal Rank Fusion, so a question matches by
   *wording* and by *meaning*.
2. **Scoped retrieval.** Metadata filters are applied **before** scoring, so
   the assistant can only ever see the slice of the corpus you allow.

It runs with **no API key, no network, and no model download**. Retrieval,
scoping, and the out-of-scope gate are entirely local. A key is only needed for
the final answer-writing step.

```bash
git clone https://github.com/beascocheatyler-star/hybrid-scoped-rag.git
cd hybrid-scoped-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/demo_hybrid.py     # guided tour, no key required
pytest                            # 267 tests, all offline
python run.py                     # web UI at http://localhost:8000
```

---

## Why hybrid

BM25 is a strong lexical baseline that fails in one predictable way: it can only
match words the question actually used. Ask *"when is the midterm?"* of a
syllabus that says *"Examination 2 is held in week 8"* and lexical retrieval has
nothing to grab.

Dense retrieval has the mirror failure. It generalises well and matches exact
rare tokens badly — course codes, order numbers, policy windows: precisely the
strings a grounded answer must quote verbatim.

Running both and fusing the rankings covers each one's blind spot:

$$\text{RRF}(c) = \sum_{r} \frac{w_r}{K + \text{rank}_r(c)}$$

**Why RRF rather than a weighted sum of raw scores.** BM25 scores are unbounded
and corpus-dependent; cosine similarities sit in $[-1, 1]$. Combining them
numerically needs a normalisation that is itself a tuning problem, and one that
shifts every time a document is added. RRF reads only *ranks*, so it is
scale-free and stable.

The cost is that RRF discards magnitude — a runaway-best match and a merely-good
one at the same rank contribute equally. That is fine for choosing *which*
passages to show the model, and **not** fine for deciding whether to answer at
all. So the two concerns are kept separate:

| Decision | Signal used |
|---|---|
| Which passages to retrieve | Fused RRF rank |
| Whether to answer at all | Calibrated BM25 × coverage |

Conflating those was the main design risk in this project, and keeping them
apart is deliberate.

### Measured contribution

From `python scripts/measure_hybrid.py` against the bundled corpora:

```
passages found by both:           46  (51%)
passages found only by lexical:    0  (0%)
passages found only by dense:     44  (49%)
```

Nearly half of all retrieved passages are ones pure BM25 would have missed
entirely.

---

## Why scoped

The predecessor project could only narrow retrieval to an explicit list of
document IDs — useful when you already know which file you want, useless
otherwise. Here every document carries arbitrary metadata, and a query may
filter on it.

```jsonc
{"course": "CS1428"}                        // exact match
{"course": ["CS1428", "CS2308"]}            // OR within a key
{"course": "CS1428", "term": "fall2025"}    // AND across keys
{"course": {"not": "CS1428"}}               // negation
{"course": {"not_in": ["CS1428"]}}          // negated set
```

Three decisions worth flagging:

- **Filtering happens before scoring, not after.** Post-filtering the top-k
  would return fewer than *k* results — or none — exactly when the filter
  matters most.
- **Unknown keys do not match.** A filter naming a key a document lacks
  excludes that document. Ignoring unknown keys would silently *widen* scope,
  the dangerous direction for a feature whose job is keeping things out.
- **Coverage is computed over the filtered subset.** The predecessor read
  corpus-wide term frequencies, so narrowing to one document still inherited
  corpus-wide coverage — a question could pass the gate on terms that appear
  only in documents the filter had just excluded. That leak is fixed here and
  pinned by `test_coverage_is_computed_over_the_scoped_subset`.

There is deliberately **no query language**. A parser for
`course = CS1428 AND term = Fall2025` is a weekend of work and a permanent
source of syntax errors in a project about retrieval. The filter is structured
JSON, which the HTTP layer already validates for free.

---

## The scope gate

Two layers, because either alone is insufficient.

**Layer 1 — retrieval gate (pre-LLM).** If the best match scores below the
relevance floor, the question is off-topic by construction. Refused with **no
token spent**. Deterministic and un-jailbreakable: no prompt talks its way past
it, because the model is never called.

Relevance is the geometric mean of calibrated BM25 and term coverage:

$$\text{relevance} = \sqrt{\text{bm25}_{\text{norm}} \times \text{coverage}}$$

The geometric mean requires *both* signals to be respectable. High BM25 with low
coverage is *"What time does the Super Bowl start?"* matching a syllabus on the
single word "time". Low BM25 with high coverage is *"Help me write a cover
letter"* sharing common words with a policy document.

**Layer 2 — grounding instruction (in-LLM).** The system prompt requires the
model to answer only from supplied context, and uploaded text is explicitly
marked untrusted data rather than instructions.

### Dense rescue

A question failing the lexical gate but scoring high densely is promoted to
`borderline` — where the model is told to verify scope before answering —
rather than refused. This is the paraphrase case.

The threshold is **measured, not guessed**:

```
in-scope questions    min 0.115  median 0.195  max 0.361
off-topic questions   min 0.038  median 0.124  max 0.206
```

The distributions overlap, so no threshold separates them cleanly. `0.25` sits
just above the off-topic maximum. A false accept costs one *guarded* LLM call,
not a wrong answer.

Rescue is suppressed when `RELEVANCE_FLOOR` is raised above the confident band:
raising the floor must never make the bot answer *more* questions.

---

## Multi-tenancy

One deployment can serve several customers, each with a corpus the others
cannot read. Off by default; existing single-corpus installs are unaffected.

```bash
MULTI_TENANT=1 TENANTS="acme:Acme University:acme-key-xxx,globex:Globex:globex-key-yyy" \
  python run.py

curl -H "X-API-Key: acme-key-xxx" localhost:8000/api/documents
```

**Tenant scope is not the same thing as the topic scope above, and is not built
the same way.** Topic scope asks "is this question on-topic?" — a tuned
judgement the user picks per request and may widen freely. Tenant scope asks
"is this data yours?" — absolute, derived from the credential, and not
influenced by any request field. The first being wrong costs an unhelpful
answer; the second being wrong is a data breach.

So tenancy is not another `ScopeFilter`. Each tenant gets its own
`DocumentStore` over its own directory, and cross-tenant chunks are not
filtered out at query time — they are **not in the index being searched**. A
filtering bug can leak; an absent document cannot.

Same question, two tenants:

```
Question: What happens if I miss an exam?
  acme (owns syllabus)     7 chunks | top hits: ['cs1428_syllabus.md']
  globex (owns returns)    5 chunks | top hits: ['returns_and_shipping.md']
```

Full design, threat model, and limits: [`docs/multi-tenancy.md`](docs/multi-tenancy.md).

---

## Honest limitations

- **The default embedder is not semantic.** `HashingEmbedder` scores character
  n-gram overlap. It buys robustness to spelling and morphology
  (`enrolment`/`enrollment`, `syllabus`/`syllabi`) which BM25 cannot do at all,
  but it will not learn that "exam" relates to "midterm". True synonym gaps stay
  unsolved — see
  `test_vocabulary_gap_is_narrowed_but_not_closed`, which pins this
  rather than hiding it. For real semantics:
  `pip install sentence-transformers && export EMBEDDING_BACKEND=sentence-transformers`
  (then re-run `measure_hybrid.py`, as the rescue floor wants to be ~0.5).
- **Off-topic refusal is 8/10, not 10/10** on the bundled set. The two misses
  land in `borderline`, where the model is instructed to verify — not in a
  confident answer.
- **Persistence is a JSON file rewritten per upload.** O(n²) bytes to ingest n
  documents, chosen so a reviewer can open and read the corpus. ~310 ms at 600
  documents. For thousands, swap in SQLite; `DocumentStore`'s interface would
  not change.
- **The index is in-memory.** Single-process. Horizontal scaling would need a
  shared vector store.
- **Tenant credentials are env-var plaintext, held in memory.** Fine for a demo
  or a small deployment; a real one needs keys hashed at rest and revocable
  without a restart. `TenantRegistry` is the seam.
- **Multi-tenancy defaults to off**, which keeps existing installs working but
  means a multi-customer deployment that forgets `MULTI_TENANT=1` gets no
  isolation while appearing to work. `/api/status` reports the live mode and
  startup logs a warning, but neither replaces checking.
- **Tenant isolation is untested under concurrency.** The 52 tenancy tests
  prove it sequentially; no test drives two tenants in parallel.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/documents` | Upload files, optionally tagged |
| `GET` | `/api/documents` | List the corpus |
| `PUT` | `/api/documents/{id}/metadata` | Re-tag without re-uploading |
| `DELETE` | `/api/documents/{id}` | Remove and de-index |
| `GET` | `/api/facets` | Available metadata keys and values |
| `POST` | `/api/search` | **Retrieval only — no LLM, no key** |
| `POST` | `/api/chat` | Ask a scoped question |
| `GET` | `/api/status` | Config, corpus stats, facets, live tenancy mode |

In multi-tenant mode every endpoint above requires `X-API-Key` and operates
only on that tenant's corpus. `/api/status` included: document counts describe
a specific corpus, so serving them unauthenticated would disclose one tenant's
holdings to anyone who asked.

`/api/search` exists so retrieval can be graded on its own terms, exposing the
per-retriever ranks RRF fused and the decision the gate would reach:

```bash
curl -X POST localhost:8000/api/search -H 'Content-Type: application/json' \
  -d '{"query": "when is the final exam", "scope": {"course": "CS1428"}}'
```

Interactive docs at `/docs`.

---

## Configuration

All optional; every value has a working default.

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Only needed to generate answers |
| `MODEL` | `claude-opus-5` | Generation model |
| `EMBEDDING_BACKEND` | `hashing` | or `sentence-transformers` |
| `EMBEDDING_DIMENSIONS` | `1024` | Hashing embedder width |
| `RRF_K` | `60` | Fusion smoothing constant |
| `LEXICAL_WEIGHT` | `1.0` | Set `0` to disable BM25 |
| `DENSE_WEIGHT` | `1.0` | Set `0` to disable vectors |
| `RELEVANCE_FLOOR` | `0.40` | Below this, refuse |
| `DENSE_RESCUE_FLOOR` | `0.25` | `1.0` disables rescue |
| `TOP_K` | `6` | Passages per answer |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | Characters |
| `DATA_DIR` | `./data` | Corpus location |
| `MULTI_TENANT` | `0` | `1` isolates tenants and requires `X-API-Key` |
| `TENANTS` | — | `id:name:key`, comma- or newline-separated |
| `DEFAULT_TENANT_ID` | `default` | Tenant used when multi-tenancy is off |

Setting `LEXICAL_WEIGHT=0` or `DENSE_WEIGHT=0` isolates either retriever, which
is how `measure_hybrid.py` attributes the contribution of each.

---

## Layout

```
app/
  config.py            Environment-driven settings
  main.py              FastAPI routes
  schemas.py           Request/response models
  core/
    embedding.py       Pluggable embedders (hashing | sentence-transformers)
    retrieval.py       HybridIndex: BM25 + dense + RRF
    metadata.py        ScopeFilter parsing and matching
    guard.py           The two-layer scope gate
    store.py           Ingest, tag, persist, reload
    chunking.py        Boundary-aware splitting
    extract.py         PDF / DOCX / CSV / JSON / MD / TXT
    persona.py         Who the bot is and what it covers
    tenancy.py         Per-tenant stores, credential resolution, isolation
web/                   Vanilla JS UI, no build step
scripts/
  demo_hybrid.py       Guided tour
  measure_hybrid.py    Reproduce every tuning number
tests/                 319 offline tests
```

## Testing

```bash
pytest                 # everything, offline, no key
pytest -q --tb=short
```

The LLM is stubbed throughout; CI sets no API key. Frontend markdown-escaping
and scroll behaviour are tested through Node harnesses when `node` is
available, and skipped cleanly when it is not.

The 52 tenancy tests are written adversarially. A test that only checks "tenant
A uploads, tenant A retrieves" passes just as happily on a system with no
isolation at all, since a single shared corpus also returns A's document to A.
Each one instead tries to *cross* the boundary — reading another tenant's
document by its real id, deleting it, retrieving its text, seeing its metadata
tags, or having its filenames surface in a refusal message — and asserts that
it fails.

## Credit

The hybrid-scoped-RAG concept came from my instructor. This repository is my
implementation of it, built as a successor to
[scoped-rag-chatbot](https://github.com/beascocheatyler-star/scoped-rag-chatbot),
which did lexical-only retrieval over a flat corpus.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

This is a copyleft license: you may use, study, modify, and redistribute this
code, but anything you distribute that is derived from it must also be released
under GPL-3.0 with its source available.

```
Copyright (C) 2026 Tyler Beascochea

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
