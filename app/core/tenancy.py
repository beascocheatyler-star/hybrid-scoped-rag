"""Tenant identity and isolation — the part that makes this a *multi-tenant* bot.

The predecessor had exactly one corpus, one persona, and one index, created as
module-level singletons at import time. Every caller saw the same knowledge
base. That is correct for a single deployment and unacceptable the moment two
customers share a process.

**The distinction that drives this whole module.** The existing ``ScopeFilter``
in ``metadata.py`` and the relevance gate in ``guard.py`` both answer *"is this
question on-topic?"* — a tuned, probabilistic judgement, where being wrong costs
an unhelpful answer. Tenant scope answers a different question, *"is this data
yours?"*, where being wrong discloses one customer's documents to another. The
two must not be implemented the same way:

====================  ==========================  ============================
                      Topic scope (``ScopeFilter``)  Tenant scope (this module)
====================  ==========================  ============================
Chosen by             the user, per request       the authenticated credential
Enforced by           relevance scoring + filter  separate store per tenant
Can be widened        yes — that is the point     never, by any request field
Cost of an error      an unhelpful answer         a data breach
====================  ==========================  ============================

So tenancy is deliberately *not* expressed as another ``ScopeFilter``. A filter
is a request parameter, and anything a request can set, a request can unset. A
caller who omits ``scope`` sees the whole corpus — exactly right for topic
scope, catastrophic for tenancy. Instead each tenant gets its **own**
``DocumentStore`` over its **own** directory. Cross-tenant retrieval is not
filtered out at query time; the other tenant's chunks are not in the index being
searched at all. There is no code path that ranks them and then discards them,
because the cheapest way to never leak a document is to never load it.

**Identity comes from the credential, never from the body.** ``resolve_tenant``
reads an API key and maps it to a tenant. It does not accept a ``tenant_id``
field, because a client that can name its own tenant has no isolation at all —
that is impersonation with extra steps.
"""

from __future__ import annotations

import hmac
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.core.embedding import Embedder
from app.core.persona import PersonaStore
from app.core.store import DocumentStore

logger = logging.getLogger(__name__)


# A tenant id becomes a directory name, so it is a path-traversal sink. Rather
# than sanitising (blacklisting ".." and separators, and hoping the list is
# complete), this allowlists the only shape that is ever legitimate. Anything
# else is rejected outright: "../other-tenant", "a/b", and an empty string all
# fail the same way. Length is capped so an id cannot blow past a filesystem's
# per-component limit and get silently truncated into a *different* tenant's
# directory.
_TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")

# Reserved because the tenant root also holds bookkeeping, and because some are
# meaningless or dangerous as directory names on common filesystems.
_RESERVED_IDS = frozenset({"con", "prn", "aux", "nul", "tenants", "shared"})


class InvalidTenantError(ValueError):
    """Raised when a tenant id is not a safe, well-formed identifier."""


class UnknownTenantError(LookupError):
    """Raised when a credential does not map to any provisioned tenant."""


def validate_tenant_id(tenant_id: str) -> str:
    """Return ``tenant_id`` if it is safe to use as a directory name.

    Raises ``InvalidTenantError`` otherwise. Call this before *any* filesystem
    use of a tenant id, including ids read back from a config file — a
    hand-edited registry is as much a trust boundary as an HTTP request.
    """
    if not isinstance(tenant_id, str):
        raise InvalidTenantError("Tenant id must be a string.")

    # Checked *before* stripping. Stripping is meant to forgive incidental
    # spaces around an id typed into a config file; it must not quietly erase a
    # control character, because that would map two different inputs
    # ("acme" and "acme\n") onto one tenant. An id that needs a newline removed
    # to become valid is not a valid id.
    if any(ch in tenant_id for ch in "\x00\r\n\t\v\f"):
        raise InvalidTenantError("Tenant id must not contain control characters.")

    candidate = tenant_id.strip(" ").lower()
    if not candidate:
        raise InvalidTenantError("Tenant id must not be empty.")

    if not _TENANT_ID.match(candidate):
        raise InvalidTenantError(
            "Tenant id {!r} is not valid. Use 1-63 characters: lowercase "
            "letters, digits, hyphen, or underscore, starting with a letter or "
            "digit.".format(tenant_id)
        )

    if candidate in _RESERVED_IDS:
        raise InvalidTenantError("Tenant id {!r} is reserved.".format(tenant_id))

    return candidate


@dataclass(frozen=True)
class Tenant:
    """One isolated customer of the service.

    Frozen because a tenant's identity must not be mutated after the request
    resolved it — rebinding ``tenant_id`` mid-request is precisely the bug this
    module exists to make impossible.
    """

    tenant_id: str
    name: str

    @property
    def display_name(self) -> str:
        return self.name or self.tenant_id


class TenantContext:
    """A tenant's own store and persona. One per tenant, never shared."""

    def __init__(self, tenant: Tenant, store: DocumentStore, personas: PersonaStore):
        self.tenant = tenant
        self.store = store
        self.personas = personas

    @property
    def tenant_id(self) -> str:
        return self.tenant.tenant_id


class TenantRegistry:
    """Maps credentials to tenants and owns each tenant's isolated state.

    Stores are created lazily and then cached: building one re-reads and
    re-indexes that tenant's corpus from disk, so doing it per request would be
    both slow and a way to lose the embedding cache. The cache is keyed by the
    validated tenant id.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        embedder: Optional[Embedder] = None,
        store_options: Optional[Dict[str, object]] = None,
    ):
        self._root = Path(data_dir)
        self._embedder = embedder
        self._store_options = dict(store_options or {})
        # RLock, not Lock: ``context_for`` holds the lock while constructing a
        # DocumentStore, and reentrancy keeps a future refactor that calls back
        # into the registry from deadlocking.
        self._lock = threading.RLock()
        self._tenants: Dict[str, Tenant] = {}
        self._keys: Dict[str, str] = {}
        self._contexts: Dict[str, TenantContext] = {}

    # --- provisioning ---------------------------------------------------

    def register(self, tenant_id: str, name: str, api_key: str) -> Tenant:
        """Provision a tenant and bind a credential to it.

        The key is held in memory only; persisting credentials is the
        deployment's job, not this module's. See ``docs/multi-tenancy.md`` for
        why this is a deliberate boundary rather than an omission.
        """
        clean_id = validate_tenant_id(tenant_id)

        if not isinstance(api_key, str) or not api_key.strip():
            raise InvalidTenantError("An API key is required to register a tenant.")
        key = api_key.strip()

        with self._lock:
            owner = self._keys.get(key)
            if owner is not None and owner != clean_id:
                # Silently rebinding a live key would hand one tenant's traffic
                # to another — the exact failure this class exists to prevent.
                raise InvalidTenantError(
                    "That API key is already bound to a different tenant."
                )

            tenant = Tenant(tenant_id=clean_id, name=name or clean_id)
            self._tenants[clean_id] = tenant
            self._keys[key] = clean_id
            return tenant

    def list_tenants(self):
        with self._lock:
            return sorted(self._tenants.values(), key=lambda t: t.tenant_id)

    # --- resolution -----------------------------------------------------

    def resolve(self, api_key: Optional[str]) -> Tenant:
        """Map a credential to its tenant, or raise ``UnknownTenantError``.

        Note what this signature does *not* take: a tenant id. Identity is
        derived from the secret alone, so there is no parameter a caller could
        set to reach another tenant's data.
        """
        if not api_key or not isinstance(api_key, str):
            raise UnknownTenantError("No API key was supplied.")

        presented = api_key.strip()
        with self._lock:
            # Compared with hmac.compare_digest rather than a dict lookup or
            # ``==``. Both of those short-circuit on the first differing byte,
            # so response time correlates with how much of a guessed prefix is
            # correct, which leaks the key a byte at a time. Every candidate is
            # compared and no loop exits early, so the work done is a function
            # of the number of tenants, not of how close the guess was.
            #
            # The cost is O(tenants) per request. At the scale this targets that
            # is nothing; a deployment with many thousands of tenants should
            # move to hashed-key lookup, which restores O(1) without
            # reintroducing the leak.
            matched: Optional[str] = None
            for known_key, tenant_id in self._keys.items():
                if hmac.compare_digest(known_key, presented):
                    matched = tenant_id

            if matched is None:
                # Deliberately uniform: distinguishing "no such key" from "key
                # for a deleted tenant" would confirm which keys once existed.
                raise UnknownTenantError("That API key is not recognised.")
            return self._tenants[matched]

    def directory_for(self, tenant_id: str) -> Path:
        """The isolated data directory for a tenant.

        Re-validates even though callers are expected to pass a validated id:
        this is the function that turns an identifier into a filesystem path,
        so it is the right place to be certain rather than trusting its callers.
        """
        clean_id = validate_tenant_id(tenant_id)
        path = (self._root / "tenants" / clean_id).resolve()

        # Defence in depth. The regex already excludes separators and dots, so
        # this cannot currently fail — it is here so that loosening the regex
        # later cannot silently reintroduce traversal.
        root = (self._root / "tenants").resolve()
        if root != path.parent:
            raise InvalidTenantError("Refusing to use an out-of-tree tenant path.")
        return path

    def context_for(self, tenant: Tenant) -> TenantContext:
        """The tenant's store and persona, building them on first use."""
        with self._lock:
            existing = self._contexts.get(tenant.tenant_id)
            if existing is not None:
                return existing

            directory = self.directory_for(tenant.tenant_id)
            directory.mkdir(parents=True, exist_ok=True)

            store = DocumentStore(
                data_dir=directory,
                embedder=self._embedder,
                **self._store_options,  # type: ignore[arg-type]
            )
            context = TenantContext(
                tenant=tenant,
                store=store,
                personas=PersonaStore(directory),
            )
            self._contexts[tenant.tenant_id] = context
            logger.info("Initialised tenant %s at %s", tenant.tenant_id, directory)
            return context


def parse_tenant_config(raw: str) -> List[Tuple[str, str, str]]:
    """Parse ``tenant_id:name:api_key`` entries into validated triples.

    Entries are separated by commas or newlines. A malformed entry raises
    rather than being skipped: quietly dropping one would start the service
    with a tenant silently missing, and the first symptom would be a customer
    unable to log in with no indication why.

    The name may contain spaces but not a colon, since the colon is the
    delimiter. Splitting from the right would be worse — an api key is far more
    likely to contain a colon than a display name is.
    """
    entries: List[Tuple[str, str, str]] = []
    if not raw or not raw.strip():
        return entries

    chunks = [c.strip() for c in raw.replace("\n", ",").split(",")]
    for chunk in chunks:
        if not chunk:
            continue
        parts = chunk.split(":", 2)
        if len(parts) != 3:
            raise InvalidTenantError(
                "Tenant entry {!r} must have the form tenant_id:name:api_key".format(
                    chunk
                )
            )
        tenant_id, name, api_key = (p.strip() for p in parts)
        if not api_key:
            raise InvalidTenantError(
                "Tenant entry {!r} has an empty API key.".format(chunk)
            )
        entries.append((validate_tenant_id(tenant_id), name or tenant_id, api_key))

    return entries
