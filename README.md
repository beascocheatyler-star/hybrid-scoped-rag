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
pytest                            # 517 tests, all offline
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

The distributions overlap, so no threshold separates them cleanly. The floor
must sit above the off-topic maximum. On the larger 160-question eval set the
off-topic maximum is 0.356, so the default is `0.36` (`scripts/measure_rescue.py`;
see `docs/scope-tuning.md`). A false accept costs one *guarded* LLM call, not a
wrong answer.

Rescue is suppressed when `RELEVANCE_FLOOR` is raised above the confident band:
raising the floor must never make the bot answer *more* questions.

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
  confident answer. On the larger 160-question eval set, 15/80 off-topic
  questions reach the model (13 as `borderline`, 2 as confident `in_scope`) and
  6/80 answerable ones are refused (`docs/eval-baseline.md`).
- **The index is in-memory, one per notebook, single-process.** Storage is
  SQLite (WAL), and each notebook's hybrid index is rebuilt from it lazily.
  Running several uvicorn workers over one `DATA_DIR` would give each worker
  its own index, which goes stale when another worker writes. Use one worker.
- **Large notebooks want NumPy.** With NumPy, search p95 is 2.2 ms at 10k and
  33 ms at 100k chunks (`scripts/bench.py search`); without it, the pure-Python
  path is about 285 ms at 10k.
- **Some source types need optional installs.** URL pages work with the
  standard library (better with `trafilatura`); PPTX, EPUB and HTML are parsed
  with the standard library. YouTube, audio, image OCR and scanned-PDF OCR need
  the packages in `requirements-extras.txt`; without them the source fails with
  an explanation. These adapters are tested with fakes, not against the real
  libraries.
- **The URL fetcher resolves DNS twice** (once to check, once to connect), so
  a DNS-rebinding attacker could race it. Private, loopback and link-local
  targets, and redirects to them, are otherwise refused.
- **Imported legacy sources have no original file.** The pre-M1 JSON store
  never kept uploads, so documents migrated from `corpus.json` can be
  searched and cited, but `GET .../file` returns 404 for them.
- **Cross-lingual retrieval is weak.** Any script now tokenizes (Chinese,
  Russian, Arabic, ...), so a question in the same language as the source
  works. A Spanish question against an English source has non-English recall@6
  of only 0.27 with hashing, 0.60 with bge-small and 0.67 with
  multilingual-e5-small (`docs/design/M3.md`).
- **Semantic recall gain is +8.8 points, short of the +15 target.** Lexical
  questions are already at 1.000; the ceiling on this set is +16.
- **Follow-ups use a heuristic.** 17/18 answerable follow-ups pass and 6/7
  off-topic follow-ups are refused, on a dataset written alongside the detector.

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
| `GET` | `/api/status` | Config, corpus stats, facets |

The table above is the v1 API. It operates on the **Default notebook** and is
unchanged for existing clients. The v2 API (M1) addresses any notebook:

| Method | Path | Purpose |
|---|---|---|
| `GET` `POST` | `/api/v2/notebooks` | List / create notebooks |
| `GET` `PATCH` `DELETE` | `/api/v2/notebooks/{nb}` | Read / rename / delete |
| `POST` | `/api/v2/notebooks/{nb}/copy` | Copy sources into a new notebook |
| `POST` | `/api/v2/notebooks/{nb}/sources` | Add files (multipart: txt, md, pdf, docx, pptx, epub, html, csv, json, images, audio) or JSON `{"kind": "text" \| "url" \| "youtube"}`; returns job ids |
| `GET` | `/api/v2/notebooks/{nb}/sources[/{id}]` | Sources with status (`queued`, `processing`, `ready`, `failed`) |
| `GET` | `/api/v2/notebooks/{nb}/sources/{id}/content?highlight=chunk` | Extracted text, chunk anchors (offsets, page, slide, timestamps, section) |
| `GET` | `/api/v2/notebooks/{nb}/sources/{id}/file` | The original upload |
| `PATCH` `DELETE` | `/api/v2/notebooks/{nb}/sources/{id}` | Rename / re-tag / remove |
| `GET` | `/api/v2/notebooks/{nb}/facets` | Metadata facets for this notebook |
| `POST` | `/api/v2/notebooks/{nb}/search` | **Retrieval + gate decision, no LLM, no key** |
| `GET` / `POST` | `/api/v2/jobs/{id}` / `.../cancel` | Job status / cancel |

Notebooks are isolated by construction: each has its own index, so IDF,
coverage, facets and duplicate detection never see another notebook. On first
boot, an existing `data/corpus.json` is imported into the Default notebook and
the original files are kept in `data/legacy-backup-<timestamp>/`.

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
| `MODEL` | `claude-opus-5-5` | Generation model (per-task: `MODEL_CHAT`, `MODEL_REWRITE`, `MODEL_STUDIO`, `MODEL_JUDGE`) |
| `LLM_PROVIDER` | `anthropic` | or `openai` for any OpenAI-compatible server (Ollama, vLLM); set `OPENAI_BASE_URL` |
| `MAX_UPLOAD_MB` / `MAX_SOURCE_WORDS` | `200` / `500000` | Per-source limits |
| `SOURCE_GUIDES` | `true` | Summary, topics and questions per source |
| `EMBEDDING_DIMENSIONS` | `1024` | Hashing embedder width |
| `RRF_K` | `60` | Fusion smoothing constant |
| `LEXICAL_WEIGHT` | `1.0` | Set `0` to disable BM25 |
| `DENSE_WEIGHT` | `1.0` | Set `0` to disable vectors |
| `RELEVANCE_FLOOR` | `0.42` | Below this, refuse (or a measured row in `calibrations`) |
| `EMBEDDING_BACKEND` | `auto` | sentence-transformers if installed, else hashing |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | e.g. `intfloat/multilingual-e5-small` for multilingual |
| `QUERY_REWRITE` | `heuristic` | Follow-up condensation: `heuristic`, `llm` or `off` |
| `MULTI_QUERY` / `RERANKER` | `false` / `off` | Context-only improvements; never change the gate |
| `LONG_CONTEXT_TOKEN_BUDGET` | `0` | Pass all in-scope chunks when they fit (off by default: cost) |
| `DENSE_RESCUE_FLOOR` | `0.36` | `1.0` disables rescue |
| `TOP_K` | `6` | Passages per answer |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `200` | Characters |
| `DATA_DIR` | `./data` | SQLite database (`hsr.sqlite3`) and stored originals |
| `JOB_WORKERS` | `2` | Background ingestion threads |
| `MAX_SOURCES_PER_NOTEBOOK` | `50` | Per notebook (the v1 Default notebook is unlimited) |
| `SINGLE_USER_MODE` | `true` | No login; everything belongs to the local user |

Setting `LEXICAL_WEIGHT=0` or `DENSE_WEIGHT=0` isolates either retriever, which
is how `measure_hybrid.py` attributes the contribution of each.

---

## Layout

```
app/
  config.py            Environment-driven settings
  main.py              FastAPI routes (v1, on the Default notebook)
  api_v2.py            /api/v2: notebooks, sources, search, jobs
  migrations/          001_initial.sql, ...
  schemas.py           Request/response models
  core/
    embedding.py       Pluggable embedders (hashing | sentence-transformers)
    retrieval.py       HybridIndex: BM25 + dense + RRF
    metadata.py        ScopeFilter parsing and matching
    guard.py           The two-layer scope gate
    db.py              SQLite connection + numbered-SQL migration runner
    notebooks.py       NotebookStore (per-notebook index) + NotebookRegistry
    jobs.py            Durable background jobs (thread pool + jobs table)
    importer.py        One-time corpus.json -> Default notebook import
    services.py        Wires db, notebooks, import and jobs together
    store.py           Legacy JSON store (kept for import and its tests)
    chunking.py        Boundary-aware splitting
    extract.py         PDF / DOCX / CSV / JSON / MD / TXT
    persona.py         Who the bot is and what it covers
examples/
  course_ta/ customer_care/   The two bundled demo corpora
  public_domain/              Constitution, Faraday, Darwin (eval corpora)
web/                   Vanilla JS UI, no build step
scripts/
  demo_hybrid.py       Guided tour
  measure_hybrid.py    Reproduce every tuning number
eval/                  Offline/live eval harness + labelled datasets
tests/                 517 offline tests (test_invariants.py pins I1–I9)
```

## Testing

```bash
pytest                 # everything, offline, no key
pytest -q --tb=short
python -m eval.run --offline                  # metrics table, free
python -m eval.run --offline --check-baseline # the CI regression gate
```

`tests/test_invariants.py` pins the scope-gate guarantees (filter before score,
zero-token refusals, monotonic floor, injection handling, ...) listed in
`docs/roadmap/notebooklm-parity.md`. CI also fails if any offline eval metric
regresses more than 2 points from `eval/baseline.json`.

The LLM is stubbed throughout; CI sets no API key. Frontend markdown-escaping
and scroll behaviour are tested through Node harnesses when `node` is
available, and skipped cleanly when it is not.

## Credit

The hybrid-scoped-RAG concept came from my instructor. This repository is my
implementation of it, built as a successor to
[scoped-rag-chatbot](https://github.com/beascocheatyler-star/scoped-rag-chatbot),
which did lexical-only retrieval over a flat corpus.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE). Derivative works
must also be distributed under GPL-3.0 with source available.
