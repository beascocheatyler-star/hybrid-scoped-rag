# Scoping: metadata filters

This document covers the *scoped* half of the project — what the filter
language does, why it stops where it does, and how to extend it.

## The shape

Every document carries a flat `Dict[str, str]` of metadata, assigned at upload
or changed later via `PUT /api/documents/{id}/metadata`. A query may supply a
filter evaluated against it.

```jsonc
{"course": "CS1428"}                        // exact match
{"course": ["CS1428", "CS2308"]}            // OR within a key
{"course": {"in": ["CS1428", "CS2308"]}}    // explicit OR
{"course": {"not": "CS1428"}}               // negation
{"course": {"not_in": ["CS1428"]}}          // negated set
{"course": "CS1428", "term": "fall2025"}    // AND across keys
```

Semantics: **AND across keys, OR within a key.** Flat, inspectable, and
sufficient for the realistic queries — course codes, terms, departments,
authors, document kinds.

The dict-with-operator form exists for a specific reason: `{"not": ...}` has to
be distinguishable from a literal value, because a bare string could
legitimately *be* the word "not".

## Three decisions that matter

### 1. Filter before scoring, never after

Post-filtering the top-k would return fewer than *k* results — or zero —
precisely when the filter is doing the most work. If the ten best matches for
"return policy" all sit in documents the filter excludes, a post-filter returns
nothing while a pre-filter returns the best in-scope matches.

`HybridIndex.search` builds the candidate list first, then scores only
candidates.

### 2. Unknown keys exclude

A filter naming a key a document lacks **excludes** that document.

The alternative — ignoring keys the document doesn't have — silently *widens*
scope. For a feature whose entire job is keeping things out, widening is the
dangerous direction. Narrowing fails visibly (you get nothing, you notice);
widening fails invisibly (you get an answer from a document that should have
been out of bounds).

Negated conditions invert this consistently: a document with no `course` key
*does* satisfy `{"course": {"not": "CS1428"}}`, because it is indeed not
CS1428.

### 3. Coverage is computed over the filtered subset

This is a bug fix relative to the predecessor project, and the subtlest part of
the system.

The scope gate scores a question as $\sqrt{\text{bm25} \times \text{coverage}}$,
where coverage is the fraction of the question's content words that appear
anywhere in scope. The predecessor computed coverage from **corpus-wide**
document frequencies. So narrowing retrieval to one document still inherited
coverage from every other document — a question could pass the gate on terms
that exist *only* in documents the filter had just excluded.

`HybridIndex._coverage` now walks the candidate set. Pinned by
`test_coverage_is_computed_over_the_scoped_subset`.

Note the deliberate asymmetry: **IDF still uses global corpus statistics.** If
term rarity depended on the filter, the same question against the same document
would score differently depending on what else happened to be in scope, and the
gate's calibrated thresholds would mean nothing. Global IDF keeps BM25 scores
comparable across filters; local coverage keeps the gate honest about what is
reachable.

## Normalisation

Keys and values are lowercased and have internal whitespace collapsed, so
`"Course"`, `"course"`, and `" COURSE "` address the same field, and `"CS1428"`
matches `"cs1428"`. Case sensitivity here would be a constant, invisible source
of empty result sets.

Limits: 64 chars per key, 256 per value, 32 keys per document. These exist to
bound the facet UI and the per-chunk memory cost, not because anything breaks
just past them.

## Why there is no query language

A parser for `course = CS1428 AND (term = Fall2025 OR term = Spring2026)` is a
weekend of work, and then a permanent source of syntax errors, error-message
design, and escaping bugs — in a project whose subject is retrieval.

The filter is structured JSON, which the HTTP layer validates for free, and
which a UI can construct from facet counts without the user typing anything.

## Extending it

**Typed values (numbers, dates).** Currently everything is a string, so
`{"year": {"gt": 2024}}` is not expressible. Adding it needs: a type registry
per key (or per-value type tags), coercion rules at ingest, comparison
semantics per type, and a decision about what `gt` means for a document whose
value won't parse. The string-plus-`in` design covers the common cases without
any of that.

**Nested boolean trees.** `Condition` is frozen and `ScopeFilter.matches` is a
simple `all(...)`, so an `Or`/`And` node type would slot in without touching
the retrieval loop. No example use case has required it.

**Chunk-level metadata.** Metadata is currently document-level, copied onto
each chunk at index time (so filtering is a local test rather than a locked
lookup). Per-chunk tags — section headings, page numbers — would need
`DocumentStore.add` to accept per-piece metadata and `update_metadata` to stop
overwriting wholesale.
