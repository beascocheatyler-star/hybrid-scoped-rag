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
    │   ├── corpus.json      ← Acme's documents, chunks, and vectors
    │   └── persona.json     ← Acme's assistant identity
    └── globex/
        ├── corpus.json
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
byte at a time.

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

**`DocumentStore` rewrites its whole corpus per upload.** That cost is now per
tenant rather than global, which helps, but the O(n²) bytes-written property
noted in `store.py` is unchanged within a single tenant.

**Contexts are cached and never evicted.** Every tenant that has made a request
holds its index in memory. Fine for tens of tenants; thousands need an LRU.

**Not tested against a concurrent adversary.** The tests prove isolation under
sequential access. `TenantRegistry` locks its mutations, but no test drives two
tenants in parallel to prove the absence of a race.

## Testing

```bash
pytest tests/test_tenancy.py tests/test_tenant_api.py -v
```

52 tests. They are written adversarially on purpose: a test that only checks
"tenant A uploads, tenant A retrieves" passes just as happily on a system with
no isolation whatsoever, because a single shared corpus also returns A's
document to A. Every test here instead tries to *cross* the boundary and
asserts that it fails — reading another tenant's document by its real id,
deleting it, retrieving its content, seeing its metadata tags, or having its
filenames appear in a refusal message.

Writing them that way was worth it immediately: the traversal cases caught a
real bug where `validate_tenant_id` stripped a trailing newline, silently
mapping `"acme\n"` and `"acme"` onto the same tenant.
