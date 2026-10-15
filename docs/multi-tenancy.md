# Multi-tenancy

How one deployment serves several customers without any of them being able to
read another's documents.

## The distinction this rests on

This project already had something called "scope" before multi-tenancy existed,
and conflating the two would be a serious mistake. They answer different
questions and must be built differently.

**Topic scope** (`ScopeFilter` in `metadata.py`, the gate in `guard.py`) asks
*"is this question on-topic?"* It is a tuned, probabilistic judgement. The user
chooses it per request and may widen it at will — that is the entire point of a
facet filter. When it is wrong, someone gets an unhelpful answer.

**Tenant scope** (`tenancy.py`) asks *"is this data yours?"* It is absolute. The
caller does not choose it; the credential does. It can never be widened by
anything in a request. When it is wrong, one customer reads another's
documents.

|                      | Topic scope                  | Tenant scope                |
| -------------------- | ---------------------------- | --------------------------- |
| Chosen by            | the user, per request        | the authenticated credential |
| Enforced by          | relevance scoring + filter   | a separate store per tenant |
| Can be widened       | yes — that is the point      | never, by any request field |
| Cost of an error     | an unhelpful answer          | a data breach               |

The practical consequence: **tenancy is not implemented as another
`ScopeFilter`.** A filter is a request parameter, and anything a request can
set, a request can unset. A caller that omits `scope` sees the whole corpus,
which is correct for topic scope and catastrophic for tenancy.

## How isolation actually works

Each tenant gets its own `DocumentStore` over its own directory:

```
data/
└── tenants/
    ├── acme/
    │   ├── corpus.db        ← Acme's documents, chunks, and vectors
    │   └── persona.json     ← Acme's assistant identity
    └── globex/
        ├── corpus.db
        └── persona.json
```

Cross-tenant retrieval is not *filtered out* at query time. The other tenant's
chunks are **not in the index being searched**. There is no code path that
ranks them and then discards them, because the cheapest way to never leak a
document is to never load it. A filtering bug can leak; an absent document
cannot.

Measured, with both tenants holding a document and asked a question only one
can answer:

```
Question: What happens if I miss an exam?
  acme (owns syllabus)     7 chunks indexed | top hits: ['cs1428_syllabus.md']
  globex (owns returns)    5 chunks indexed | top hits: ['returns_and_shipping.md']
```

Globex returns its own best-effort match rather than the syllabus, because the
syllabus is not in its index to return.

## Identity comes from the credential

`TenantRegistry.resolve` takes an API key and nothing else. Note what its
signature does *not* accept: a tenant id. A client that can name its own tenant
has no isolation at all — that is impersonation with extra steps.

Endpoints reach a corpus only through the `current_tenant` dependency, which is
what makes the guarantee structural rather than a convention. There is no
module-level store left to reach for, so an endpoint cannot accidentally skip
authentication; it would have nothing to operate on.

Key comparison uses `hmac.compare_digest` rather than a dict lookup or `==`.
Both of those short-circuit on the first differing byte, so response time
correlates with how much of a guessed prefix is correct, which leaks the key a
byte at a time. Keys are compared as UTF-8 bytes, since `compare_digest`
rejects non-ASCII `str` with a `TypeError` — which previously turned any
such header into a 500 carrying the presented key in the traceback.

The honest limit: `compare_digest` is constant-time only for equal-length
inputs, so this still leaks the *length* of valid keys. Closing that needs
hashing every candidate to a fixed width, which is the same change a large
deployment wants anyway for O(1) lookup.

## Configuration

Off by default, so existing single-deployment installs keep working unchanged.

```bash
MULTI_TENANT=1
TENANTS="acme:Acme University:acme-key-xxx,globex:Globex Retail:globex-key-yyy"
```

Then pass the key on every request:

```bash
curl -H "X-API-Key: acme-key-xxx" localhost:8000/api/documents
```

With `MULTI_TENANT=0` (the default) no key is needed and the service exposes a
single corpus exactly as before.

## Storage

Each tenant's corpus is a SQLite database at `tenants/<id>/corpus.db`.

The previous design was a single JSON file rewritten in full on every mutation.
It was atomic per write — temp file, `fsync`, rename — and still lost data,
because atomicity is not the same as isolation. Each worker process held its
own copy of the whole corpus, so the second save overwrote the first with a
snapshot that never contained the first worker's document. Reproduced with two
processes before any of this was written:

```
uploaded: ['a.md', 'b.md']  (both returned success)
on disk : ['a.md']
LOST    : ['b.md']
```

**The bug was not "JSON" — it was read-modify-write of the entire dataset.** No
lock fixes that: an in-process lock cannot see another process, and a file lock
would serialise whole-corpus rewrites into a bottleneck while still losing data
on a crash between read and write. The fix is to make each mutation touch only
its own rows inside a transaction the database arbitrates.

Verified against a real four-worker uvicorn server, twelve concurrent uploads:

```
HTTP responses   : [200] | count: 12
documents on disk: 12 (expected 12)
orphaned chunks  : 0
integrity_check  : ok
```

Three defects surfaced while stress-testing this, all invisible to
single-process tests and all only reachable when workers start at the same
instant:

- `PRAGMA busy_timeout` was set *after* `journal_mode=WAL`, leaving the one
  statement most likely to contend unprotected. Workers died at startup with
  `database is locked`.
- The schema-version row was inserted after a read — check-then-act. Several
  workers all saw no row, all inserted, and the losers died on the primary key.
- `busy_timeout` does not cover the lock that switching journal mode needs, so
  WAL has to be retried explicitly. Failure is logged rather than fatal: WAL is
  a concurrency optimisation, and crashing a worker over a performance setting
  is worse than running in the slower mode.

Existing `corpus.json` files are imported automatically on first boot and kept
as a backup. An upgrade that deletes the only copy of the data before anyone
has confirmed the new path works is not a migration.

## Known limits

Stated plainly, because a security feature whose boundaries are vague invites
being trusted where it shouldn't be.

**Credentials live in memory and in an env var.** Suitable for a demo or a
small deployment. A real one needs keys hashed at rest, rotatable, and revocable
without a restart. `TenantRegistry` is the seam for that: replace its two dicts
with a database and nothing above it changes.

**The permissive default is a real risk.** Deploying for several customers and
forgetting `MULTI_TENANT=1` yields no isolation while appearing to work. Two
mitigations: `/api/status` reports the live mode, and startup logs a warning
when tenants are configured but the flag is off. Neither is a substitute for
checking.

**No per-tenant resource limits.** One tenant can consume disk and CPU to the
detriment of others. Isolation here is about *confidentiality*, not fairness.

**No per-tenant disk quota.** A tenant's database grows without limit.

**Contexts are cached and never evicted.** Every tenant that has made a request
holds its index in memory. Fine for tens of tenants; thousands need an LRU.

**Index freshness lags across workers.** Document durability is solved — see
[the storage notes](#storage) — but each worker builds its **search index** in
memory at startup. A document uploaded through worker 1 is on disk immediately
and will be loaded by every worker at its next start; until then worker 2
cannot retrieve it. The same question can be answered or refused depending on
which worker handles it.

This is a staleness window, not data loss, and it closes at restart. Closing it
properly means workers detecting each other's writes and rebuilding, which is a
separate piece of work from the durability fix and deliberately not bundled
with it.

**Concurrency is tested, but not exhaustively.** `test_tenant_regressions.py`
covers racing cold starts, parallel persona writes, and the cold-start blocking
hazard. It does not model an adversary deliberately timing requests to find a
window.

## What the audit found

The first implementation passed 52 adversarial tests and still contained an
authentication bypass. That is worth recording, because it is the same lesson
as the eval in the predecessor project: *passing the tests you thought to write
is not evidence that the tests covered the right things.*

**An auth bypass (critical).** The default tenant was registered with the
literal key `"__single_tenant__"` so that one code path could serve both modes.
That string is published in the source. In multi-tenant mode, anyone could send
it as `X-API-Key` and get full read, write, and delete on the default tenant's
corpus. The fix is structural rather than a filter: `register_keyless` creates
a tenant with **no entry in the key table at all**, so `resolve` cannot match it
however it is called. A sentinel value that is checked is still a credential; the
only safe sentinel is the absence of one.

**A silent corpus merge (high).** `validate_tenant_id` lowercases, and
`register` overwrote rather than rejecting. So `ACME:Acme Corp:key-a` and
`acme:Acme Rival:key-b` both registered, both resolved, and both received the
*same* `DocumentStore`. Each could read and delete the other's documents, with
no error and nothing in the log. `register` now rejects a duplicate id.

**A shared persona object (high).** `PRESETS` holds mutable `Persona` objects
and `apply_preset` stored them by reference. Two tenants applying the same
preset were aliased onto one object, so an in-place edit by one appeared in the
other's persona *and in its LLM system prompt* — and permanently polluted
`PRESETS` for every tenant afterwards, including the unauthenticated
`/api/persona/presets`. Now deep-copied.

**A 500 instead of a 401 (medium).** `hmac.compare_digest` raises `TypeError`
on non-ASCII `str`. Any UTF-8 key crashed the request and logged a traceback
containing the presented key, defeating the uniform-response property the code
was written to guarantee. Keys are now compared as UTF-8 bytes.

**Non-atomic persona writes (medium).** `PersonaStore.set` used `write_text`,
which truncates before writing, with no lock. Interleaved writes could produce
invalid JSON, which `_load` then silently reset to the default persona at the
next boot — losing a tenant's name, topic, and style. It now uses the same
temp-file + `fsync` + atomic-rename pattern as `DocumentStore._save`, and adopts
the new persona only *after* the write succeeds.

**A cold-start liveness hazard (medium).** `context_for` held the global
registry lock across store construction, which reads the corpus and may
re-embed every chunk. One large tenant's first request blocked every other
tenant's, and because the endpoints are sync `def` they occupy threadpool
workers while blocked — enough of them would stall the service entirely.
Construction now happens under a per-tenant lock with a double check.

## Testing

```bash
pytest tests/test_tenancy.py tests/test_tenant_api.py \
       tests/test_tenant_regressions.py -v
```

74 tests. They are written adversarially on purpose: a test that only checks
"tenant A uploads, tenant A retrieves" passes just as happily on a system with
no isolation whatsoever, because a single shared corpus also returns A's
document to A. Every test here instead tries to *cross* the boundary and
asserts that it fails — reading another tenant's document by its real id,
deleting it, retrieving its content, seeing its metadata tags, or having its
filenames appear in a refusal message.

Writing them that way was worth it immediately: the traversal cases caught a
real bug where `validate_tenant_id` stripped a trailing newline, silently
mapping `"acme\n"` and `"acme"` onto the same tenant.
