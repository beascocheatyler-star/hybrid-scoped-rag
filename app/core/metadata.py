"""Scope filters: the *scoped* half of hybrid scoped RAG.

The predecessor project could only narrow retrieval to an explicit list of
document ids. That works when the user already knows which file they want, and
not at all otherwise — "only search the Fall 2025 syllabi" had no expression.

Here every document carries an arbitrary ``Dict[str, str]`` of metadata, and a
query may supply a *filter* that is evaluated against it. Filtering happens
**before** scoring, not after: a post-filter over the top-k would return fewer
than k results (or none) whenever the best matches fall outside the scope, which
is precisely when the filter matters most.

Design constraints that shaped this:

* **No query language.** A parser for ``course = CS1428 AND term = Fall2025``
  would be a weekend of work and a permanent source of syntax errors in a
  project whose point is retrieval, not parsing. The filter is structured data
  (JSON), which the HTTP API already validates for free.
* **Values are strings.** Typed values (numbers, dates) would need coercion
  rules, a type registry per key, and comparison semantics. Strings plus the
  ``in`` operator covers the realistic cases — course codes, terms, authors,
  departments, tags — without any of that. ``docs/scoping.md`` records what a
  typed extension would require.
* **Unknown keys do not match.** A filter naming a key a document lacks
  excludes that document. The alternative (ignore unknown keys) silently widens
  the scope, which is the dangerous direction for a feature whose job is to
  keep things out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

# Metadata keys and values are normalised so that "Course", "course", and
# " COURSE " address the same field, and "CS1428" matches "cs1428". Case
# sensitivity here would be a constant, invisible source of empty result sets.
MAX_KEY_LENGTH = 64
MAX_VALUE_LENGTH = 256
MAX_KEYS_PER_DOCUMENT = 32


class InvalidFilterError(ValueError):
    """Raised when a scope filter is malformed."""


def normalize_key(key: str) -> str:
    return " ".join(str(key).strip().lower().split())


def normalize_value(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def normalize_metadata(raw: Optional[Mapping[str, object]]) -> Dict[str, str]:
    """Coerce user-supplied metadata into a clean ``{str: str}`` mapping.

    Non-string values are stringified rather than rejected: a user tagging a
    document ``{"year": 2025}`` through the JSON API means something obvious,
    and failing the upload over it would be hostile.
    """
    if not raw:
        return {}
    if not isinstance(raw, Mapping):
        raise InvalidFilterError("metadata must be an object of key/value pairs")

    cleaned: Dict[str, str] = {}
    for key, value in raw.items():
        norm_key = normalize_key(key)
        if not norm_key:
            continue
        if len(norm_key) > MAX_KEY_LENGTH:
            raise InvalidFilterError(
                "metadata key '{}...' exceeds {} characters".format(
                    norm_key[:20], MAX_KEY_LENGTH
                )
            )
        if isinstance(value, bool):
            # str(True) is "True"; lowercase reads better and matches how a
            # user would write the filter.
            norm_value = "true" if value else "false"
        elif isinstance(value, (list, tuple, set)):
            raise InvalidFilterError(
                "metadata value for '{}' must be a single value, not a list. "
                "Use several keys, or a comma-joined string.".format(norm_key)
            )
        else:
            norm_value = normalize_value(value)
        if len(norm_value) > MAX_VALUE_LENGTH:
            raise InvalidFilterError(
                "metadata value for '{}' exceeds {} characters".format(
                    norm_key, MAX_VALUE_LENGTH
                )
            )
        cleaned[norm_key] = norm_value

    if len(cleaned) > MAX_KEYS_PER_DOCUMENT:
        raise InvalidFilterError(
            "a document may carry at most {} metadata keys".format(
                MAX_KEYS_PER_DOCUMENT
            )
        )
    return cleaned


@dataclass(frozen=True)
class Condition:
    """One ``key`` must match one of ``values`` (OR within a key)."""

    key: str
    values: frozenset
    negated: bool = False

    def matches(self, metadata: Mapping[str, str]) -> bool:
        actual = metadata.get(self.key)
        if actual is None:
            # A document without the key cannot satisfy a positive condition.
            # For a negated condition it trivially does: a document with no
            # `course` key is indeed "not course=CS1428".
            return self.negated
        hit = actual in self.values
        return (not hit) if self.negated else hit

    def describe(self) -> str:
        joined = " or ".join(sorted(self.values))
        return "{} {} {}".format(self.key, "is not" if self.negated else "is", joined)


@dataclass
class ScopeFilter:
    """A conjunction of conditions. Empty means "search everything".

    AND across keys, OR within a key. That combination covers the queries this
    is for — ``course=CS1428 AND term in (fall2025, spring2026)`` — while
    staying a flat, inspectable structure. Arbitrary nested boolean trees were
    considered and rejected: they need a real parser, and no example use case
    required one.
    """

    conditions: List[Condition] = field(default_factory=list)
    # Explicit document allowlist, preserved from the predecessor's API so
    # existing clients keep working.
    doc_ids: Optional[Set[str]] = None

    @property
    def is_empty(self) -> bool:
        return not self.conditions and self.doc_ids is None

    def matches(self, doc_id: str, metadata: Mapping[str, str]) -> bool:
        if self.doc_ids is not None and doc_id not in self.doc_ids:
            return False
        return all(c.matches(metadata) for c in self.conditions)

    def describe(self) -> str:
        """Human-readable summary, surfaced in the API so a refusal is explicable."""
        parts = [c.describe() for c in self.conditions]
        if self.doc_ids is not None:
            parts.append("limited to {} selected document(s)".format(len(self.doc_ids)))
        return " and ".join(parts) if parts else "the whole knowledge base"

    @classmethod
    def parse(
        cls,
        raw: Optional[Mapping[str, object]] = None,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> "ScopeFilter":
        """Build a filter from the JSON shape accepted by the API.

        Accepted forms per key::

            {"course": "CS1428"}                     exact match
            {"course": ["CS1428", "CS2308"]}         match any  (OR)
            {"course": {"in": [...]}}                explicit any
            {"course": {"not": "CS1428"}}            negation
            {"course": {"not_in": [...]}}            negated any

        The dict-with-operator form exists because ``{"not": ...}`` has to be
        distinguishable from a literal value; a bare string could legitimately
        be the word "not".
        """
        conditions: List[Condition] = []

        if raw:
            if not isinstance(raw, Mapping):
                raise InvalidFilterError("filter must be an object of key/value pairs")
            for key, spec in raw.items():
                norm_key = normalize_key(key)
                if not norm_key:
                    raise InvalidFilterError("filter contains an empty key")
                conditions.append(cls._parse_condition(norm_key, spec))

        ids = set(doc_ids) if doc_ids else None
        return cls(conditions=conditions, doc_ids=ids)

    @staticmethod
    def _parse_condition(key: str, spec: object) -> Condition:
        negated = False
        values: Iterable

        if isinstance(spec, Mapping):
            operators = {normalize_key(k): v for k, v in spec.items()}
            if len(operators) != 1:
                raise InvalidFilterError(
                    "filter for '{}' must specify exactly one operator "
                    "(in, not, not_in); got {}".format(key, sorted(operators) or "none")
                )
            op, operand = next(iter(operators.items()))
            if op in {"in", "any"}:
                values = operand if isinstance(operand, (list, tuple, set)) else [operand]
            elif op in {"not", "ne", "not_in"}:
                negated = True
                values = operand if isinstance(operand, (list, tuple, set)) else [operand]
            elif op in {"is", "eq"}:
                values = operand if isinstance(operand, (list, tuple, set)) else [operand]
            else:
                raise InvalidFilterError(
                    "unknown operator '{}' for key '{}'. Supported: in, not, "
                    "not_in, is.".format(op, key)
                )
        elif isinstance(spec, (list, tuple, set)):
            values = spec
        else:
            values = [spec]

        normalized = frozenset(
            normalize_value(v) for v in values if normalize_value(v) != ""
        )
        if not normalized:
            raise InvalidFilterError(
                "filter for '{}' has no usable values".format(key)
            )
        return Condition(key=key, values=normalized, negated=negated)


def collect_facets(
    metadatas: Iterable[Mapping[str, str]]
) -> Dict[str, Dict[str, int]]:
    """Count value frequencies per key across the corpus.

    Powers the UI's scope picker: the user should choose from the values that
    actually exist rather than guessing and getting an empty result set.
    """
    facets: Dict[str, Dict[str, int]] = {}
    for meta in metadatas:
        for key, value in meta.items():
            facets.setdefault(key, {})
            facets[key][value] = facets[key].get(value, 0) + 1
    return facets
