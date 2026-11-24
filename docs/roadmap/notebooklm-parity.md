# Roadmap: NotebookLM-class feature parity

> Status: **proposal / implementation brief**. Nothing in this document is
> implemented yet. It is written as an engineering brief (usable directly as a
> prompt for a coding agent) and pins the scope-safety invariants that any
> implementation must preserve.

# ROLE
You are a senior Python/FastAPI and vanilla-JavaScript engineer. You are extending an existing, well-tested open-source repository so that it reaches feature parity with Google NotebookLM (renamed "Gemini Notebook" in July 2026). Work incrementally and in milestones. Keep the existing test suite green. Never weaken the scope-enforcement guarantees described below.

# REPOSITORY
https://github.com/beascocheatyler-star/hybrid-scoped-rag (branch main, HEAD 2adfca9). Clone it. Read README.md, docs/architecture.md, docs/scoping.md and docs/scope-tuning.md before writing any code.

# CURRENT SYSTEM (accurate as of HEAD 2adfca9; verify as you read)

Stack
- Python 3.9–3.12 (CI matrix in .github/workflows/tests.yml).
- FastAPI, uvicorn, python-multipart, pydantic v2, anthropic SDK, pypdf, python-docx (requirements.txt). Dev: pytest, httpx.
- Frontend: vanilla JS/HTML/CSS with NO build step (web/index.html, web/app.js, web/style.css). It includes its own markdown renderer, which escapes input first (web/app.js around L193–330). XSS safety is tested by tests/markdown_harness.js.
- Config: environment variables with defaults (app/config.py). A tiny .env loader lives in app/__init__.py.
- LLM: Anthropic Messages API through client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=..., messages=..., output_config={"effort": EFFORT}). Defaults: MODEL=claude-opus-5, EFFORT=medium, MAX_TOKENS=2000 (app/core/engine.py ~L357–364).
- Entry point: python run.py (uvicorn app.main:app). API docs at /docs.

Modules
- app/main.py — routes:
  - POST/GET/DELETE /api/documents
  - PUT /api/documents/{id}/metadata
  - GET /api/facets
  - GET /api/supported-types
  - POST /api/search (retrieval and gate decision only, no LLM)
  - GET/PUT /api/persona, POST /api/persona/preset, GET /api/persona/presets
  - POST /api/chat
  - GET /api/status
  - GET / (serves web/index.html; /static is the mounted web/ directory)
- app/schemas.py — pydantic models: ChatRequest{message, history[], doc_ids?, scope?}, ChatResponse{answer, in_scope, decision, citations[], retrieval_score, dense_score, rescued_by_dense, scope_description, documents_in_scope, reason, usage, model}, SearchRequest/Response, etc.
- app/core/extract.py — file bytes to text for .txt .md .markdown .pdf (pypdf, inserts "[page N]" markers, no OCR) .docx .csv .json.
- app/core/chunking.py — greedy paragraph, then sentence, then word packing to CHUNK_SIZE=1200 characters with CHUNK_OVERLAP=200 tail overlap. Chunks carry NO page or character offsets.
- app/core/embedding.py — Embedder interface:
  - HashingEmbedder (default): signed char 3/4-gram hashing, 1024 dimensions, L2-normalised, NOT semantic.
  - SentenceTransformerEmbedder("all-MiniLM-L6-v2"): optional, selected with EMBEDDING_BACKEND=sentence-transformers. Falls back to hashing if it fails to load.
- app/core/retrieval.py — HybridIndex (in-memory, RLock):
  - BM25 with k1=1.5, b=0.75. ASCII tokenizer regex [a-z0-9][a-z0-9'_-]* plus English stopwords.
  - IDF uses GLOBAL corpus statistics, deliberately.
  - Dense scoring: brute-force pure-Python cosine; similarity <= 0 is dropped.
  - Fusion: Reciprocal Rank Fusion sum(w/(RRF_K+rank)) with RRF_K=60 and LEXICAL_WEIGHT/DENSE_WEIGHT. Candidate depth is max(top_k*4, 20). Ties are broken by lexical score, then dense score.
  - Each SearchHit carries: score (fused, ordering only), normalized (BM25 / max possible for the query, 0..1), coverage (fraction of query terms present in the FILTERED candidate set), dense_score, lexical_rank, dense_rank.
  - The scope filter is applied BEFORE scoring.
- app/core/metadata.py — ScopeFilter:
  - Document metadata is Dict[str,str], normalised (lowercase, collapsed whitespace).
  - Filter JSON forms: exact value; list (OR); {"in":[...]}; {"not":v}; {"not_in":[...]}. AND across keys. Optional doc_ids allow-list.
  - A key the document lacks means the document does NOT match.
- app/core/guard.py — classify() returns one of: no_corpus, empty_scope, small_talk (regex), meta (regex), out_of_scope, borderline, in_scope.
  - relevance = sqrt(normalized * coverage) of hits[0], the fused top hit.
  - RELEVANCE_FLOOR=0.40 and CONFIDENT_RELEVANCE=0.50.
  - Dense rescue: below the floor, if the best dense score >= DENSE_RESCUE_FLOOR (0.25), promote to borderline (never to in_scope). Rescue is disabled when the floor is above 0.50, so raising the floor never answers more.
  - refusal_message() produces templated refusals with NO LLM call.
- app/core/engine.py — ChatEngine.answer():
  - Calls store.count_matching(scope), then index.search(raw question, top_k=TOP_K=6, scope), then classify.
  - If the decision is not allowed, returns a deterministic refusal.
  - Otherwise _generate() builds a system prompt (grounding rules, persona topic, active scope description, strict-scope rule, a prompt-injection rule saying CONTEXT is untrusted data, and up to 40 document names) and a user turn ("CONTEXT PASSAGES ... [Source n] document | passage | relevance | tags ... QUESTION ... instruction"). Borderline and dense-rescued questions get stricter "verify scope first" instructions.
  - History: the client sends turns. The server keeps the last 8, prose only, and drops a trailing user turn.
  - Citations returned = ALL retrieved hits, whether or not the model used them. The model is told to cite by document NAME, e.g. "(per syllabus.pdf)".
  - stop_reason == "refusal" is handled. SDK errors are translated into readable messages.
- app/core/store.py — DocumentStore:
  - Extract, then SHA-256 duplicate check, then chunk, then embed, then index.
  - Persists EVERYTHING (document metadata, chunks, embedding vectors) by rewriting data/corpus.json on every change (atomic temp file + fsync). Corpus versions 1 and 2 are tolerated on load.
  - Vectors are cached and keyed by "embedder_name:dimensions".
  - Original uploaded files are NOT kept: data/uploads/ is created but unused.
- app/core/persona.py — one global Persona{name, topic, style, greeting, suggested_questions} persisted in data/persona.json. Presets: customer_care, course_ta, generic.
- web/ — sidebar with:
  - upload dropzone and a free-text tags field ("course=CS1428, term=fall2025")
  - document list with delete
  - metadata facet chips that build the scope
  - persona editor
  The chat pane shows an "Answered from documents"/"Outside scope" tag and a collapsible "N source passages" list with a lexical/dense/both retriever tag. Chat history is held in the browser and capped at 16 messages.
- scripts/ — demo_hybrid.py, demo_scope.py, measure_hybrid.py, measure_scope.py (these reproduce all tuning numbers), build_md_test.py, build_scroll_test.py.
- tests/ — 267 offline tests with a stubbed LLM, covering retrieval, guard, metadata, API, corrupt state, embedding cache, and the frontend markdown/scroll harnesses run through node or JavaScriptCore.
- examples/ — course_ta/cs1428_syllabus.md and customer_care/returns_and_shipping.md.

Known defects to fix first
1. CI is red. tests/markdown_harness.js and tests/scroll_harness.js call print(), which is undefined in Node, and Node is preferred when it is present. Result: 2 failed, 265 passed.
2. README says "MIT" but LICENSE is GPL-3.0. Do not choose for the owner. Add a TODO in the PR description asking which license is intended, and make the README match LICENSE in the meantime.
3. The gate reads BM25 'normalized' of the fused top hit. A dense-only top hit has normalized=0, which can sink relevance even when a lexically strong hit sits at rank 2.
4. The tokenizer is ASCII-only, so non-Latin questions produce no tokens and are refused as "no searchable terms".
5. Retrieval uses only the raw current question, so follow-up questions retrieve poorly.

# GOAL
Turn this into a self-hostable, NotebookLM-class research and study notebook:
- notebooks containing many source types
- grounded chat with passage-level, verifiable, click-through citations
- persistent chats and notes
- a Studio panel of generated outputs (reports, study aids, mind maps, audio overviews, then slides and video)
- multi-user sharing
The project's distinctive strengths must remain intact and visible: hybrid retrieval, filter-before-score scoping, and the deterministic pre-LLM scope gate. They are the differentiator against NotebookLM, so do not "simplify them away".

# NON-NEGOTIABLE INVARIANTS (scope-gate preservation)
Write a dedicated test module, tests/test_invariants.py, that pins every one of these. Run it at every milestone.
I1. Filtering happens BEFORE scoring. A source outside the active notebook, the source selection, or the metadata filter is never a candidate for retrieval, reranking, long-context stuffing, Studio generation, audio scripts or citation verification.
I2. Notebook isolation is absolute. No query, cache, IDF statistic, facet list, suggestion or artifact may read data from another notebook. (Inside a notebook, keep global-IDF-within-the-notebook semantics.)
I3. Coverage is computed over the filtered subset only.
I4. Unknown metadata keys EXCLUDE documents (they never widen scope).
I5. Off-topic questions are refused BEFORE any LLM call, with zero tokens spent. Tests assert the LLM stub is never called. This must also hold for query rewriting: if rewriting uses an LLM, it is allowed only after a cheap pre-gate on the raw question passes OR the question is classified as a follow-up of an in-scope conversation. The final gate decision is always made on retrieval results computed deterministically, never on LLM output.
I6. Monotonic floor: raising RELEVANCE_FLOOR never increases the set of answered questions. Dense rescue (and any reranker or semantic signal) can only promote to borderline and never to in_scope, unless a new calibration is measured and committed via scripts/measure_*.py with the numbers in docs/scope-tuning.md.
I7. Uploaded or imported content is untrusted data. Keep and extend the prompt-injection rule to every prompt (chat, Studio, audio script, mind map, quiz). Add tests with an injected "ignore previous instructions" source and assert that outputs neither follow it nor leak the system prompt.
I8. Empty-scope and no-corpus remain distinct decisions with distinct messages.
I9. /api/search (or its notebook-scoped successor) keeps working with no API key, so retrieval can be graded for free.

# TECH STACK GUIDANCE (match the existing stack)
- Keep FastAPI + pydantic v2 + the anthropic SDK, and keep Python 3.9 compatibility (CI matrix 3.9/3.11/3.12). Use `from __future__ import annotations` as the code already does.
- Storage: stdlib sqlite3 in WAL mode, behind the existing DocumentStore-style interface (new: app/core/db.py plus repositories).
  - Store vectors as float32 BLOBs and load them per notebook into a NumPy matrix.
  - Optional sqlite-vec or hnswlib for notebooks with more than 50k chunks.
  - Write a numbered-SQL-file migration runner.
  - Write an importer that converts an existing data/corpus.json (versions 1 and 2) and data/persona.json into a "Default notebook" on first boot, and keeps a backup of the old files.
- Background jobs: a jobs table plus an in-process ThreadPoolExecutor worker (default). Optional arq/RQ adapter if REDIS_URL is set. Jobs survive restarts: on boot, re-queue "running" jobs as "queued".
- New optional dependencies go in requirements-extras.txt and are imported lazily with clear ExtractionError/feature-unavailable messages, exactly as pypdf and python-docx are handled today: numpy, sentence-transformers, trafilatura, youtube-transcript-api, faster-whisper, pytesseract + Pillow, pdf2image, python-pptx, ebooklib, pydub, argon2-cffi, sse-starlette (or a hand-rolled StreamingResponse). The core install must still boot and pass tests with requirements.txt + numpy only.
- Frontend: keep vanilla JS with no build step. Split web/app.js into ES modules (web/js/*.js) loaded with type="module". Keep escape-first markdown rendering and extend markdown_harness.js for every new renderer (citation chips, notes, mind map labels). Do not introduce React or a bundler.
- LLM: add app/core/llm.py with an LLMClient protocol (complete, stream, json-structured output via a tool/JSON schema). Anthropic is the default implementation; also provide an OpenAI-compatible implementation, which covers Ollama, vLLM and others. Per-task models: MODEL_CHAT, MODEL_REWRITE, MODEL_STUDIO, MODEL_JUDGE, each defaulting to MODEL. Keep output_config effort support for Anthropic.
- TTS: app/core/tts.py with a TTSProvider protocol. Implementations:
  - Piper (local, offline, default if installed)
  - OpenAI TTS
  - ElevenLabs
  - Google Cloud TTS
  Configured with TTS_PROVIDER and per-voice environment variables. Stitch audio with pydub/ffmpeg into MP3.
- STT (audio sources and interactive mode): faster-whisper locally, or an OpenAI-compatible transcription API.

# DATA MODEL (SQLite; add created_at/updated_at to all tables)
```
users(id, email UNIQUE, password_hash, display_name)
sessions(id, user_id, expires_at)
notebooks(id, owner_id, title, description, emoji, chat_config_json, overview_json, created_at)
memberships(notebook_id, user_id, role CHECK in ('owner','editor','viewer'))
share_links(id, notebook_id, access CHECK in ('restricted','link','public'), chat_only BOOL, token)
sources(id, notebook_id, kind CHECK in ('file','url','youtube','text','audio','image','note'), title, original_name, mime, sha256, size_bytes, uri, status CHECK in ('queued','processing','ready','failed'), error, metadata_json, guide_json, char_count, created_at)
  - UNIQUE(notebook_id, sha256): duplicates are per notebook, preserving today's duplicate rejection.
source_files(source_id, path)   -- originals stored under DATA_DIR/notebooks/{nb}/sources/{source_id}/
chunks(id, source_id, notebook_id, position, text, start_char, end_char, page, slide, t_start_ms, t_end_ms, section_path, token_count)
chunk_vectors(chunk_id, embedder_signature, dim, vector BLOB)
chats(id, notebook_id, user_id, title, selected_source_ids_json, scope_json)
messages(id, chat_id, role, content, decision, reason, retrieval_json, citations_json, usage_json, model, feedback)
notes(id, notebook_id, author_id, title, body_md, citations_json, from_message_id, is_source BOOL)
artifacts(id, notebook_id, created_by, type, params_json, status, prompt_used, content_json, file_path, source_ids_json, error)
study_progress(user_id, artifact_id, item_id, state, attempts)
jobs(id, notebook_id, type, payload_json, status, progress, error, started_at, finished_at)
calibrations(embedder_signature, relevance_floor, dense_rescue_floor, measured_at, report_path)
```

# API (version under /api/v2; keep /api/* v1 routes working against the "Default notebook" until a deprecation milestone)
Auth: POST /api/v2/auth/register, /login, /logout; GET /api/v2/me. Add a SINGLE_USER_MODE=true default, so the app still runs locally with no login, exactly like today.
Notebooks: GET/POST /notebooks; GET/PATCH/DELETE /notebooks/{nb}; POST /notebooks/{nb}/copy (copies sources and artifacts, not chats or notes, as NotebookLM does).
Sources:
- POST /notebooks/{nb}/sources (multipart files; JSON {kind:url|youtube|text, ...}), returns job ids
- GET /notebooks/{nb}/sources
- GET /notebooks/{nb}/sources/{id} (includes guide)
- GET /notebooks/{nb}/sources/{id}/content?highlight=chunk_id (text with anchors)
- GET /notebooks/{nb}/sources/{id}/file (original)
- PATCH (title, metadata) and DELETE /notebooks/{nb}/sources/{id}
- GET /notebooks/{nb}/facets
Retrieval: POST /notebooks/{nb}/search {query, source_ids?, scope?, top_k?} — no LLM, returns hits plus the gate decision (successor to /api/search).
Chat:
- POST /notebooks/{nb}/chats; GET /notebooks/{nb}/chats; GET/DELETE /chats/{id}
- POST /chats/{id}/messages (non-streaming)
- POST /chats/{id}/messages/stream (SSE events: decision, rewrite, retrieval, token, citations, done, error)
- POST /messages/{id}/feedback
- POST /messages/{id}/save-to-note
Notes: CRUD /notebooks/{nb}/notes; POST /notes/{id}/convert-to-source.
Studio: POST /notebooks/{nb}/artifacts {type, params, source_ids?}, returns a job; GET /notebooks/{nb}/artifacts; GET/DELETE /artifacts/{id}; GET /artifacts/{id}/download?format=md|docx|pdf|csv|json|mp3|pptx|png|svg; PATCH /artifacts/{id}/progress (study state).
Jobs: GET /jobs/{id}; GET /jobs/{id}/events (SSE); POST /jobs/{id}/cancel.
Sharing: GET/PUT /notebooks/{nb}/members; PUT /notebooks/{nb}/share; GET /s/{token} (shared view; chat-only mode hides the sources and studio UI).
Overview: GET /notebooks/{nb}/overview (summary plus suggested questions, regenerated when sources change).

# CITATION REQUIREMENTS (apply to chat AND every text Studio output)
C1. The context block numbers passages [1]..[k]. The prompt requires a bracketed marker after every sentence that states a fact from the sources, e.g. "The final is in week 16 [2].", and allows multiple markers like [1][3]. Sentences that do not need a citation (transitions, refusals) carry none.
C2. A server-side parser extracts the markers and maps each one to chunk_id, source_id, page/slide/timestamp, start_char and end_char. Markers that point to a passage that was not provided are removed and logged as "hallucinated_citation".
C3. Verification: for each (sentence, cited chunk) pair, compute a support score (lexical overlap of content words with the chunk, plus embedding cosine). Below SUPPORT_FLOOR (calibrated in eval), mark the citation "weak" and the sentence "unsupported". The UI shows unsupported sentences with a subtle warning style. Optional MODEL_JUDGE verification behind CITATION_JUDGE=true.
C4. Returned citations = cited passages only, in marker order, each with a quote excerpt (the best-matching sentence span inside the chunk) for hover preview. Keep the existing retrievers, lexical_rank and dense_rank fields in a separate "retrieval" debug object.
C5. Clicking a citation opens the source viewer at the exact span, highlighted. PDFs show the page number. YouTube and audio sources show and link a timestamp.
C6. Saving a message to a note preserves its citations, and they stay clickable.

# MILESTONES (do them in order; each one is a separate PR-sized commit series, with tests, docs and a CHANGELOG entry)

M0 — Baseline and hygiene (P0)
- Fix the frontend harness for Node: inject `globalThis.print = (...a) => console.log(...a);` in scripts/build_md_test.py and scripts/build_scroll_test.py bundles. CI must be green on 3.9, 3.11 and 3.12.
- README/LICENSE consistency (see Known defects #2).
- Add eval/ (see EVALUATION) with the bundled corpora. Record baseline numbers in docs/eval-baseline.md.
- Add tests/test_invariants.py covering I1–I9 against the current code.
Acceptance: pytest fully green, including the frontend harnesses with Node 20+. `python -m eval.run --offline` prints the baseline table.

M1 — SQLite storage, notebooks, originals, jobs (P0)
- Implement the schema, migrations, repositories and the corpus.json/persona.json importer.
- HybridIndex becomes per-notebook: a registry keyed by notebook_id, loaded lazily, rebuilt from SQLite.
- Keep originals on disk.
- Ingestion runs as jobs with status updates.
- Map the existing single-corpus /api/* routes to the Default notebook.
Acceptance:
- Uploading the 2 example files into notebook A and only one into notebook B means search in B never returns A's chunks (test).
- IDF and facets are per notebook (test).
- Upload time no longer grows with corpus size: benchmark 600 small uploads, each under 50 ms after embedding.
- All 267 legacy tests pass, adapted only where a route moved, and every adaptation is listed in the PR.

M2 — Anchored ingestion and new source types (P0/P1)
- The chunker emits start_char/end_char against the stored extracted text, page (PDF: per-page extraction instead of inline markers, plus offsets), slide (PPTX), t_start_ms/t_end_ms (YouTube and audio transcripts) and section_path (markdown/DOCX headings). Keep chunk size and overlap semantics.
- New extractors:
  - URL: trafilatura main-text extraction, title, canonical URL. SSRF guard blocks private, loopback and link-local addresses. Maximum 10 MB.
  - YouTube transcript via youtube-transcript-api, with timestamps.
  - Pasted text.
  - Audio via faster-whisper, with timestamps.
  - Images via Tesseract OCR.
  - Scanned PDF OCR fallback when pypdf yields no text.
  - PPTX via python-pptx (slide text plus notes).
  - EPUB via ebooklib.
- Limits configurable per source: MAX_SOURCE_WORDS=500000, MAX_UPLOAD_MB=200 (matching NotebookLM), MAX_SOURCES_PER_NOTEBOOK=50 by default.
- Source Guide job per source: summary (150–250 words with citations), 5–10 key topics, 3–5 suggested questions, stored in guide_json. Uses map-reduce for long sources.
- Auto-labelling: when a notebook has 5 or more sources, cluster sources into topic labels (LLM over guide summaries) and store them as metadata key "label". The user can edit labels. This reuses the existing metadata and facet machinery.
Acceptance:
- For each new source kind, a fixture ingests and its chunks round-trip: text[start:end] == chunk text.
- A PDF citation reports the correct page in a test.
- A URL to 127.0.0.1 is rejected.
- Missing optional dependencies produce a clear per-source error, not a crash.

M3 — Retrieval quality (P0)
- Unicode tokenizer (regex \w+ with re.UNICODE, lowercased, NFKC-normalised) and optional stopword lists per language. Keep the English behaviour identical on the existing tests.
- Default embedder: if sentence-transformers is installed, use EMBEDDING_MODEL (default BAAI/bge-small-en-v1.5; document intfloat/multilingual-e5-small for multilingual use). Otherwise use hashing. Re-embed automatically when the signature changes (existing behaviour).
- NumPy vectorised cosine per notebook.
- Query understanding:
  - (a) a deterministic follow-up detector (pronouns or deictic words plus short length, or an explicit "it/that/they" with a prior in-scope turn)
  - (b) condensation into a standalone query, via LLM (MODEL_REWRITE) or a heuristic fallback that concatenates the last in-scope user question
  - (c) optional multi-query (up to 3 sub-queries), fused with RRF
  Respect invariant I5.
- Optional reranker (RERANKER=cross-encoder) over the top 30. It reorders context only; gate inputs are unchanged.
- Gate fix: relevance uses the hit with the max normalized BM25 among the top-k (or the BM25 top-1), not hits[0]. Add a regression test for the dense-only-top-hit case.
- Calibration: extend scripts/measure_hybrid.py and measure_scope.py to write a calibration row per embedder signature. The guard reads floors from calibrations when present, else from env defaults.
- Long-context mode: if the selected sources fit LONG_CONTEXT_TOKEN_BUDGET (default 150k tokens), pass full anchored text instead of top-k. The gate still runs first on retrieval results. Citations use the same marker scheme, with passages defined as chunks.
Acceptance:
- Retrieval recall@6 improves on eval by at least 15 points over the M0 baseline with the semantic embedder installed, and does not regress with hashing.
- Gate false-accept rate does not increase. False-refuse rate drops.
- A follow-up dataset (at least 15 two-turn conversations) answers at least 80% in scope.
- The non-Latin question test passes.
- Every invariant test still passes.

M4 — Grounded answers with verifiable citations, persistent chat, streaming (P0)
- Implement C1–C6 (citation requirements).
- Persistent chats and messages per user. Feedback. Save to note.
- SSE streaming. Refusals stream as a single "decision" plus "done" event with no LLM call.
- Chat configuration per notebook (replaces the global persona and keeps the presets as templates): goal/role text, style (default | learning_guide | custom), response length (short | default | long), output language. The "learning_guide" style asks guiding questions and checks understanding before giving full answers.
- Notebook overview: summary of all sources plus 3–5 suggested questions. Regenerate when the source set changes, as a job.
Acceptance:
- On eval, citation precision is at least 0.85 and citation recall at least 0.75 (definitions in EVALUATION). Hallucinated-marker rate is 0 after parsing.
- Every streamed answer ends with a citations event.
- A chat persists across reloads, and the user can delete it.
- Viewers in a shared notebook cannot read other users' chats (test).

M5 — Notebook UI (P0)
- Home page: notebook grid (create, rename, delete, copy).
- Notebook page: three panels.
  - Sources: add-sources dialog (files / URL / YouTube / paste), per-source status, checkbox selection plus select-all, label groups, metadata facet chips (kept), source viewer drawer with highlight, Source Guide.
  - Chat: streaming, inline [n] chips with hover card (quote, source, page/timestamp), stop/regenerate/copy/save-to-note/thumbs, suggested questions, a "Why this answer?" expander showing the gate decision, relevance, coverage, dense score, rescue flag and per-retriever ranks (keep today's transparency).
  - Studio: output cards, a customise dialog, a job list with progress, and a notes board.
- Responsive: panels become tabs under 900 px. Keyboard accessible. ARIA labels on all new controls.
- Extend the markdown/XSS harness to the citation-chip renderer and the notes renderer.
Acceptance: manual QA checklist in docs/qa.md covering all flows. Harness tests cover the new renderers. Lighthouse accessibility score of at least 90 on the notebook page.

M6 — Studio text artifacts and study tools (P1)
- Generation pipeline: map step (per-source Source Guide, reused) then reduce step (artifact-specific prompt), producing JSON validated by pydantic, then rendering. Every artifact stores prompt_used, the source_ids snapshot and its citations. Outputs do not change when sources change later; a "stale" badge shows if the source set has changed.
- Types and params:
  - report: format in {briefing, study_guide, faq, timeline, blog_post, custom}, plus custom_prompt, language and length. Markdown with [n] citations. Export to MD, DOCX (python-docx) and PDF (optional weasyprint, or fall back to HTML).
  - data_table: user describes the rows and columns. CSV plus a citations sheet (CSV with row,col to citation).
  - mind_map: JSON {id,label,children[],citations[]}, depth up to 4, 20–80 nodes. Rendered as a collapsible, zoomable inline SVG with vanilla JS. Clicking a node sends "Tell me about <label>" into chat with the same source selection. Download as SVG or PNG.
  - flashcards: difficulty, quantity (fewer=10, standard=20, more=40), focus prompt. Items {id, front, back, citations}. Study UI with flip, Got it / Missed it, progress saved per user, retake all or missed only, and an "Explain" action that opens chat grounded on the card's citations.
  - quiz: difficulty, quantity, types subset of {multiple_choice, multi_select, short_answer, fill_blank}. Items carry answer key, rationale and citations. Short answers are graded by MODEL_JUDGE against the cited passage. After finishing, show results and offer to ask the chat for weak areas.
- Each artifact is a job, cancellable, with progress events.
Acceptance:
- Every artifact item has at least one valid citation into the selected sources (automated check).
- Artifacts generated with a source deselected never cite it (invariant I1 test).
- Mind map node count and depth are within bounds.
- Quiz JSON validates.
- Study progress persists across reloads.

M7 — Audio Overview (P1)
- Script generator (MODEL_STUDIO), with format in {deep_dive (2 hosts), brief (1 speaker, <=2 min), critique (2 hosts), debate (2 hosts, opposing)}, length in {short, default, long} (target word counts 600/1500/3000 for deep_dive), language, and focus prompt.
  - Output JSON: [{speaker:"A"|"B", text, citations[]}].
  - The hosts must stay faithful to the sources (an objective reflection, not opinion), must not invent facts, and must name sources naturally.
  - Run a faithfulness pass that drops or rewrites lines with unsupported claims (reuse C3).
- TTS: render each line with the configured voices, insert 150–300 ms gaps, normalise loudness, export MP3 plus a JSON transcript with per-line timestamps.
- Player: play/pause, ±15 s, speed 0.75–2x, transcript synced to the current line, a citation chip on each line opening the source viewer, and download.
- Stretch (P2) "interactive mode": a Join button pauses playback. Push-to-talk records audio, then STT, then a grounded chat answer in the host's voice (gate applies), then playback resumes.
Acceptance:
- A 2-source fixture produces an MP3 whose duration is within ±25% of the length target, using a stub TTS that writes silence of the right length in CI.
- The transcript lines carry valid citations.
- An off-topic focus prompt (for example "talk about football" on the syllabus notebook) is refused by the gate before any LLM call.

M8 — Multi-user, sharing, quotas (P1/P2)
- Auth (argon2), sessions (HttpOnly, SameSite=Lax cookies), CSRF token on mutating routes, per-route role checks (viewer: read sources, artifacts and own chats; editor: plus add/remove sources, generate, share; owner: plus delete notebook, manage members).
- Share links: restricted, link or public; chat-only flag (document the same caveat NotebookLM documents: chat-only hides the UI, it is not an access control unless CHAT_ONLY_STRICT=true, which also blocks the source and artifact APIs for that token).
- Usage analytics per notebook: users per day and queries per day for the last 7 days.
- Quotas: MAX_NOTEBOOKS_PER_USER, MAX_SOURCES_PER_NOTEBOOK, daily artifact limits. Rate limiting per IP and per user.
Acceptance: an authorization matrix test (role × route) is fully enumerated. Public share works without login for artifacts only. SINGLE_USER_MODE still works with no login.

M9 — Breadth (P2; only after M0–M8 are green)
- Discover sources: SearchProvider protocol (Brave, SerpAPI, Tavily or SearXNG via env). Fast mode returns 10 candidates with title, snippet, URL and a "why relevant" line, which the user imports selectively. Deep mode is an iterative plan, search, read, note and report loop (max N pages, budgeted) that produces a cited report plus importable sources.
- Slide deck: outline JSON, then PPTX via python-pptx (detailed | presenter formats), with speaker notes carrying citations.
- Video Overview: slide images (render PPTX to PNG via LibreOffice headless if available, else HTML-to-PNG), narration via TTS, assembled with ffmpeg into MP4 with captions (SRT).
- Infographic: a template-driven SVG (orientation landscape/portrait/square, detail concise/standard/detailed).
- PWA manifest and offline audio playback.
- Optional sandboxed code runner for CSV/XLSX analysis and chart PNGs (subprocess, no network, CPU, memory and time limits).

# EVALUATION HARNESS (build in M0; extend each milestone)
- Layout:
  - eval/datasets/*.jsonl
  - eval/run.py (CLI: --offline uses the stub LLM and the deterministic judge; --live uses real models)
  - eval/metrics.py
  - eval/report.py (writes docs/eval-latest.md)
- Datasets (hand-written over the bundled examples plus at least 3 new public-domain corpora you add under examples/, e.g. a public-domain textbook chapter, a government policy PDF and a lecture transcript):
  - retrieval.jsonl: {notebook, question, relevant_chunk_ids|relevant_spans}. At least 120 questions, including 30 paraphrases with no lexical overlap and 15 non-English questions.
  - gate.jsonl: {notebook, scope, question, expected: answer|refuse}. At least 80 answerable and 80 off-topic, including adversarial lexical-overlap cases like those in docs/scope-tuning.md.
  - followups.jsonl: multi-turn conversations with an expected standalone intent.
  - answers.jsonl: {question, reference_facts[]}, used for faithfulness and citation scoring.
  - injection.jsonl: sources containing injected instructions, plus probes.
- Metrics:
  - recall@k, MRR, nDCG@10
  - gate false-accept and false-refuse rates
  - citation precision = valid supporting citations / all citations
  - citation recall = reference facts stated with at least one valid citation / reference facts stated
  - unsupported-sentence rate
  - faithfulness (judge)
  - injection resistance (must be 100%)
  - latency p50/p95
  - tokens per answer
- CI gate: the offline eval runs in GitHub Actions and fails if any metric regresses more than 2 points from the committed baseline, or if any invariant or injection case fails.

# TESTING REQUIREMENTS
- Keep everything runnable offline with no API key. Stub the LLM and TTS (deterministic fakes in tests/fakes.py).
- Every milestone adds unit tests (pure functions: tokenizer, chunk anchors, citation parser/verifier, filters), API tests (httpx TestClient) and at least one end-to-end flow test.
- Property-based tests (hypothesis, dev dependency) for chunk offset round-trips and for ScopeFilter parse/matches.
- Frontend: extend the Node harnesses for every renderer that writes innerHTML. Add a small Playwright smoke test (optional dev dependency, skipped if not installed) for upload, ask, hover citation and open the source viewer.
- Migrations: test upgrading a v1 and a v2 corpus.json fixture into SQLite.
- Performance: a benchmark script (scripts/bench.py) for ingest throughput and search latency at 10k and 100k chunks. Target search p95 under 150 ms at 100k chunks with NumPy on a laptop.

# NON-GOALS
- Do not replicate Google-account features (Drive sync, Play Books, Gemini app integration, SynthID).
- Do not train or fine-tune models.
- Do not replace FastAPI or add a JS framework or bundler.
- Do not remove the deterministic gate, the per-retriever transparency, /api/search, or the scope filter language. Do not add a free-text query language for filters (the repo deliberately avoids one).
- Do not send user documents to any third-party service unless that provider is explicitly configured. The default local stack (hashing or local sentence-transformers, Piper, faster-whisper, Tesseract) must work fully offline except for the chosen LLM.

# HOW TO WORK
1. Start by reading the code and running pytest. Reproduce "2 failed, 265 passed" and fix it (M0) before anything else.
2. For each milestone: write a short design note in docs/design/Mx.md (data, API, prompts, risks), then the tests, then the code, then the docs. Commit in small, reviewable steps with conventional commit messages.
3. After every milestone, run: pytest -q; python -m eval.run --offline; python scripts/measure_hybrid.py; python scripts/measure_scope.py. Paste the results into the milestone summary.
4. Keep the README honest, in the existing style. Update "Honest limitations" with anything still missing or measured as weak.
5. When a decision trades scope safety against recall, choose safety and document the trade in docs/scope-tuning.md with numbers.
6. If an optional dependency is unavailable in your environment, implement the adapter behind its protocol with a fake for tests, and say so explicitly. Never claim a feature works without a passing test.
7. At the end of each milestone, output: what changed (files), how to run it, test and eval results, known gaps, and the next milestone plan.
