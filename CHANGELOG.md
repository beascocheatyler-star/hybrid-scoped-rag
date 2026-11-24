# Changelog

## Unreleased — M2: anchored ingestion and new source types

### Added
- Anchored chunks: every chunk stores `start_char`/`end_char` into the stored
  extracted text (`text[start:end] == chunk`), plus `page` (PDF), `slide`
  (PPTX), `t_start_ms`/`t_end_ms` (audio, YouTube) and `section_path`
  (Markdown, DOCX, EPUB and HTML headings). Chunking decisions are unchanged
  (same words per chunk), so retrieval and every eval metric are identical.
- Structured extraction (`extract_doc`): PDF per-page text with offsets
  (no more inline `[page N]` markers), DOCX headings, Markdown headings.
- New source kinds: URL (SSRF-guarded fetch, optional trafilatura), YouTube
  transcripts, pasted text, audio (faster-whisper or an OpenAI-compatible
  STT endpoint), images (Tesseract OCR), scanned-PDF OCR fallback, PPTX
  (slides and speaker notes), EPUB, HTML. PPTX/EPUB/HTML need no extra package.
- Limits: `MAX_UPLOAD_MB=200`, `MAX_SOURCE_WORDS=500000`.
- Source Guides (summary with passage citations, topics, questions), as a job
  after each ingest: LLM map-reduce when configured, deterministic extractive
  fallback otherwise. Hallucinated citation markers are dropped.
- Auto-labels at 5 or more sources, written to metadata `label` (so they
  become facet chips); a label the user edited is never overwritten.
- `app/core/llm.py`: `LLMClient` protocol with Anthropic (default; structured
  JSON via `output_config.format`, streaming, server-side refusal fallback)
  and OpenAI-compatible (Ollama, vLLM, ...) implementations; per-task models.
- `GET .../sources/{id}/content?highlight=<chunk_id>` returns anchors.
- `requirements-extras.txt`; `hypothesis` dev dependency with property tests.

### Changed
- Default `MODEL` is now `claude-opus-5-5` (effort `medium`).
- Two legacy tests used `.png` as an example of an unsupported file type;
  images are now supported, so they use `.exe` instead. These are the only
  legacy-test adaptations.


## Unreleased — M1: SQLite storage, notebooks, originals, jobs

### Added
- SQLite storage (WAL, foreign keys) with a numbered-SQL migration runner
  (`app/core/db.py`, `app/migrations/001_initial.sql`). Vectors are stored as
  float32 BLOBs keyed by embedder signature.
- Notebooks: `NotebookStore` gives each notebook a private hybrid index, so
  IDF, coverage, facets and duplicate detection are per notebook by
  construction. `NotebookRegistry` creates, lists, updates, copies and deletes
  notebooks. A Default notebook always exists.
- Original uploads and extracted text are kept on disk per source.
- Durable background ingestion jobs (`app/core/jobs.py`): progress,
  cancellation at stage checkpoints, and re-queue of interrupted work on boot.
- One-time importer of a legacy `corpus.json` (v1 or v2) and `persona.json`
  into the Default notebook, with a timestamped backup of the old files.
- `/api/v2`: notebooks CRUD and copy; sources upload (multipart files or
  pasted text), list, get, content, original file, patch, delete; facets;
  notebook-scoped search (no LLM, no key); jobs get/cancel; `GET /me`.
- Settings: `SINGLE_USER_MODE`, `JOB_WORKERS`, `MAX_SOURCES_PER_NOTEBOOK`.
- `scripts/bench.py` (ingest throughput, search latency).
- Tests: `tests/test_m1_storage.py`, `tests/test_api_v2.py`. Invariant I2 now
  tests real notebook isolation.

### Changed
- The v1 `/api/*` routes are served by the Default notebook's SQLite store.
  No legacy test needed adapting.
- Per-upload ingest cost no longer grows with corpus size: about 3.4 ms per
  upload at 600 documents (hashing, 1024d), against about 310 ms with the JSON
  store.
- The eval harness builds its notebooks through the SQLite path; every metric
  is identical to the M0 baseline.


## Unreleased — M0: baseline and hygiene

### Fixed
- Frontend markdown/XSS and scroll harnesses failed under Node (`print` is not
  defined), which turned CI red whenever Node was present (2 failed, 265
  passed). The test bundlers now shim `print` for Node, and CI pins Node 20.
- README said the project is MIT; `LICENSE` is GPL-3.0. README now matches
  `LICENSE`.

### Changed
- `DENSE_RESCUE_FLOOR` default raised from 0.25 to 0.36. On the 160-question
  eval gate set, 0.25 rescued 5 off-topic and 0 answerable questions; 0.36
  cuts false accepts from 25.0% to 18.8% with no change in false refusals
  (`scripts/measure_rescue.py`, `docs/scope-tuning.md`). Raising this floor can
  only reduce the set of answered questions.

### Added
- `tests/test_invariants.py`: pins scope-gate invariants I1–I9 (filter before
  score, isolation, filtered coverage, unknown keys exclude, zero-token
  refusals, monotonic floor, injection handling, empty-scope vs no-corpus,
  key-free `/api/search`).
- `tests/fakes.py`: `FakeLLM`, a deterministic client that records calls.
- `eval/`: offline/live evaluation harness (`python -m eval.run --offline`),
  span-labelled datasets (retrieval, gate, follow-ups, answers, injection),
  metrics, and markdown report. Baseline in `eval/baseline.json` and
  `docs/eval-baseline.md`.
- CI step: offline eval fails the build on a >2-point regression or any
  injection-resistance failure.
- `docs/design/M0.md`.
- Three public-domain eval corpora in `examples/public_domain/` (US
  Constitution, Faraday's candle lecture I, Darwin ch. III), from Wikisource.
- Datasets at roadmap size: 125 retrieval questions (30 paraphrases, 15
  non-English), 160 gate questions (80/80), 15 follow-ups, 16 answer sets,
  12 injection cases. `tests/test_eval_datasets.py` checks every label against
  its source text.
- `scripts/measure_rescue.py`: rescue-floor sweep over the eval gate set.
