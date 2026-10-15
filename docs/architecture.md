# Architecture

## Request flow

```
POST /api/chat  { "message": "...", "scope": {"course": "CS1428"} }
        │
        ▼
  ScopeFilter.parse(scope)
        │
        ▼
  HybridIndex.search(question, scope)
        │
        ├─ 1. FILTER   candidates = chunks matching the scope   ← before scoring
        │
        ├─ 2. SCORE    BM25 ranking        ─┐
        │              dense ranking       ─┤
        │                                   │
        └─ 3. FUSE     RRF over both lists ─┘
            │   each hit carries:
            │     • fused RRF score        (ordering only)
            │     • bm25 normalized 0–1    (the gate reads this)
            │     • coverage 0–1           (over the *filtered* subset)
            │     • dense cosine 0–1
            ▼
  guard.classify()
        │
        ├─ corpus empty? ───────────────────► refuse, no LLM call
        ├─ scope matches 0 docs? ───────────► empty_scope, no LLM call
        ├─ greeting? ───────────────────────► small_talk → LLM
        ├─ "what can you do?" ──────────────► meta → LLM
        ├─ relevance < 0.40 ────────────────┐
        │     └─ dense ≥ 0.25? ─────────────┤──► borderline → LLM + strict verify
        │        else ──────────────────────┴──► refuse, no LLM call   ← free
        ├─ 0.40 ≤ relevance < 0.50 ─────────► borderline → LLM + verify instruction
        └─ relevance ≥ 0.50 ────────────────► in_scope → LLM
        │
        ▼
  engine._generate()
        │   system prompt: grounding + scope + active filter + injection defence
        │   user turn:     [Source 1..k] passages + question + instruction
        ▼
  Claude  ──►  answer + citations (with per-retriever provenance) + usage
```

Two key properties:

- **Off-topic questions terminate before the LLM call** — refusal is free,
  instant, and immune to prompt manipulation.
- **Scope is enforced at retrieval, not by instruction.** A filtered-out
  document is never a candidate, so no prompt can coax the model into citing
  it: the text was never in the context window.

---

## Modules

| File | Responsibility |
|---|---|
| `app/main.py` | HTTP routes, upload handling, error translation |
| `app/config.py` | Environment-driven settings with defaults |
| `app/schemas.py` | Pydantic request/response contracts |
| `app/core/extract.py` | File bytes → plain text, per format |
| `app/core/chunking.py` | Text → overlapping passages on natural boundaries |
| `app/core/embedding.py` | Pluggable embedders: hashing (default) or sentence-transformers |
| `app/core/retrieval.py` | `HybridIndex`: BM25 + dense + RRF fusion, scoped |
| `app/core/metadata.py` | `ScopeFilter` parsing, normalisation, matching |
| `app/core/guard.py` | The scope decision |
| `app/core/engine.py` | Prompt construction, Claude call, citation assembly |
| `app/core/persona.py` | Persona config + presets, persisted |
| `app/core/store.py` | Document lifecycle and JSON persistence |

---

## Key design decisions

### Two independent scope layers

A prompt instruction alone is negotiable — users can argue with it, and
instruction text inside an uploaded document can try to override it. A retrieval
gate the model never sees cannot be argued with.

Conversely, retrieval alone is too blunt: a question can share genuine
vocabulary with the corpus without being answerable from it. The two layers
cover each other's blind spots.

### Scoring with a geometric mean

`√(bm25 × coverage)` rather than either signal alone, or their average. The
mirror-image failure modes are documented with measurements in
[`scope-tuning.md`](scope-tuning.md). The geometric mean collapses toward zero if
either factor is weak, which is the desired bias.

### An explicit uncertainty band

Where the in-scope and off-topic score distributions overlap, no threshold is
correct. Treating that band as `borderline` and delegating to the model — with
the retrieved text in hand — converts a guaranteed misclassification into a
judgment made with evidence.

### Hybrid retrieval, with fusion separated from gating

BM25 matches wording; dense vectors match meaning. Each fails where the other
succeeds, so both run and their *rankings* are fused with Reciprocal Rank
Fusion.

RRF rather than a weighted sum of raw scores: BM25 scores are unbounded and
corpus-dependent while cosine similarities sit in [-1, 1], so combining them
numerically needs a normalisation that is itself a tuning problem — and one
that shifts every time a document is added. RRF reads only ranks, so it is
scale-free and stable.

The cost is that RRF discards magnitude. That is acceptable for *ordering*
passages and unacceptable for *gating*, which needs to know how good the best
match is in absolute terms. So the two are kept strictly separate: fusion
decides which passages to show, calibrated BM25 × coverage decides whether to
answer at all. Conflating them was the main design risk in this project.

### The embedder is pluggable, and the default is honest about itself

The retrieval score powers the gate, so it runs on **every** question — making
cost the dominant constraint. The default `HashingEmbedder` is character-n-gram
hashing: no dependencies, no download, microseconds per chunk.

It is **not** a semantic model, and the code says so rather than implying
otherwise. What it buys is robustness to spelling and morphology
(`enrolment`/`enrollment`, `syllabus`/`syllabi`) which BM25 cannot do at all.
True synonym gaps need `EMBEDDING_BACKEND=sentence-transformers`.

The fusion architecture is identical either way, which is the point of the
interface — and is why the vocabulary-gap test asserts the gap is *narrowed*
rather than claiming it is closed.

### Vectors are cached, keyed by vector space

Chunk embeddings are persisted alongside the text, keyed by
`embedder.name:dimensions`. Switching backends invalidates the cache and
re-embeds at boot.

This matters more than a speed optimisation: vectors from two different
embedders are not comparable. A stale cache would not crash — it would rank
confidently and wrongly, the worst failure mode available.

### Chunk overlap

Chunks are split on paragraph, then sentence, then word boundaries, and each
chunk carries the tail of its predecessor. Without overlap, a policy spanning a
boundary gets split mid-statement and neither half answers the question.

### Prompt-injection defence

Uploaded documents are untrusted input. The system prompt states that context
passages are data, never instructions, and names the specific patterns to ignore
("ignore your rules", "you are now…", "reveal your prompt"). Passages are
additionally wrapped in explicit delimiters so the boundary is unambiguous.

On the frontend, all document-derived text is HTML-escaped before any markdown
rendering, so a document cannot inject script into the reviewer's browser.

### Persistence as an inspectable database

`corpus.db` holds the documents and their chunks, one database per tenant. A
reviewer can open it with any SQLite client and see exactly what the bot knows.
The hybrid index is rebuilt from it at startup — cheap at this scale, and it
avoids a binary index format nobody can inspect.

Every mutation is a transaction touching only its own rows. This replaced a
single JSON file rewritten in full on each write, which was atomic per write
and still lost documents when two worker processes ingested at once: each held
its own copy of the whole corpus, so the second save erased the first. The
lesson generalises — *atomicity is not isolation*, and a read-modify-write of
an entire dataset is unsafe however carefully the write itself is performed.
See [`multi-tenancy.md`](multi-tenancy.md#storage).

---

## Concurrency

`HybridIndex` and `DocumentStore` guard their mutable state with an `RLock`.
Embedding happens *outside* the index lock in `add_chunks`: with the
sentence-transformers backend it is by far the slowest step, and holding the
lock through it would stall every concurrent chat.
FastAPI serves requests from a thread pool, and uploads mutate the index while
chat requests read it, so this is load-bearing rather than defensive.

Across *processes*, those locks are worthless — a lock in one worker is
invisible to another. Durability is handled a level down instead, by SQLite
transactions, which is why the persistence layer and not a mutex is what makes
`--workers N` safe. What remains per-process is the in-memory index: a worker
only retrieves what it loaded at startup plus its own writes.

---

## Cost profile

| Event | LLM cost |
|---|---|
| Upload a document | none — extraction, chunking, and embedding are all local |
| Off-topic question | **none** — refused by the gate |
| Any `/api/search` call | **none** — retrieval only, never calls the LLM |
| Question outside the active scope | **none** — `empty_scope`, refused |
| Greeting / meta question | one call, minimal context |
| In-scope question | one call: system prompt + `TOP_K` passages + history |

Conversation history is replayed as prose only; prior turns' retrieved context
is not resent, which keeps multi-turn conversations from growing quadratically.
