"""FastAPI application: upload and tag documents, then chat within their scope."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.core.embedding import get_embedder
from app.core.engine import ChatEngine, LLMUnavailableError
from app.core.extract import SUPPORTED_EXTENSIONS, ExtractionError
from app.core.guard import classify
from app.core.metadata import InvalidFilterError, ScopeFilter
from app.core.persona import PRESETS, Persona, PersonaStore
from app.core.store import DocumentStore, DuplicateDocumentError
from app.core.tenancy import (
    InvalidTenantError,
    Tenant,
    TenantContext,
    TenantRegistry,
    UnknownTenantError,
    parse_tenant_config,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentModel,
    FacetsResponse,
    MetadataRequest,
    PersonaModel,
    PresetRequest,
    RetrievalInfo,
    SearchHitModel,
    SearchRequest,
    SearchResponse,
    StatusResponse,
    UploadResponse,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

settings = get_settings()
embedder = get_embedder(settings.embedding_backend, settings.embedding_dimensions)

# One registry owns every tenant's store and persona. The embedder is shared
# deliberately: it is stateless and read-only (a hashing function or a frozen
# model), so sharing it leaks nothing between tenants while avoiding a
# per-tenant copy of a model that can be hundreds of megabytes. Everything that
# holds *data* — corpus, index, persona — is per tenant.
registry = TenantRegistry(
    data_dir=settings.data_dir,
    embedder=embedder,
    store_options={
        "chunk_size": settings.chunk_size,
        "overlap": settings.chunk_overlap,
        "rrf_k": settings.rrf_k,
        "lexical_weight": settings.lexical_weight,
        "dense_weight": settings.dense_weight,
    },
)

# In single-tenant mode this is the only tenant, and it needs no credential.
# Serving both modes through one registry means the endpoints below have no
# tenancy branching in them at all: there is exactly one way to reach a store,
# so there is no unguarded path to forget about.
#
# Registered *without* a key. An earlier version gave it the sentinel
# "__single_tenant__", which was an authentication bypass: the string is public
# in the source, so in multi-tenant mode anyone could send it as X-API-Key and
# get full read/write/delete on this tenant. A keyless tenant has no entry in
# the key table, so resolve() cannot reach it however it is called.
#
# Misconfiguration here is fatal by design — a service that cannot establish
# who its tenants are must not start. But this runs at import, before the
# exception handlers below are registered, so a raw exception would reach the
# operator as a bare traceback. Each failure is re-raised with a message that
# names the setting to fix.
try:
    DEFAULT_TENANT = registry.register_keyless(settings.default_tenant_id, "Default")
except InvalidTenantError as exc:
    raise RuntimeError(
        "DEFAULT_TENANT_ID={!r} is not a valid tenant id: {} "
        "Use 1-63 characters of lowercase letters, digits, hyphen, or "
        "underscore.".format(settings.default_tenant_id, exc)
    ) from exc

try:
    _tenant_entries = parse_tenant_config(settings.tenant_config)
except InvalidTenantError as exc:
    raise RuntimeError("TENANTS is malformed: {}".format(exc)) from exc

for _tid, _name, _key in _tenant_entries:
    if _tid == DEFAULT_TENANT.tenant_id:
        # Otherwise a configured tenant would collide with the default and the
        # two would share one corpus. register() would also catch this, but
        # naming DEFAULT_TENANT_ID points at the actual cause.
        raise RuntimeError(
            "Tenant id {!r} in TENANTS collides with DEFAULT_TENANT_ID={!r}. "
            "Ids are compared case-insensitively. Rename the tenant or set "
            "DEFAULT_TENANT_ID to something else.".format(
                _tid, settings.default_tenant_id
            )
        )
    try:
        registry.register(_tid, _name, _key)
    except InvalidTenantError as exc:
        raise RuntimeError("TENANTS entry {!r} rejected: {}".format(_tid, exc)) from exc

_configured_tenants = [
    t for t in registry.list_tenants() if t.tenant_id != DEFAULT_TENANT.tenant_id
]
if _configured_tenants and not settings.multi_tenant:
    logger.warning(
        "%d tenant(s) are configured but MULTI_TENANT is not enabled, so every "
        "request will be served the single default corpus and the tenant "
        "credentials will be ignored. Set MULTI_TENANT=1 to enforce isolation.",
        len(_configured_tenants),
    )
elif settings.multi_tenant:
    logger.info(
        "Multi-tenant mode: %d tenant(s) registered; each request must carry a "
        "valid X-API-Key.",
        len(_configured_tenants),
    )

engine = ChatEngine(settings=settings)


def _warn_if_multiprocess() -> None:
    """Note the remaining multi-worker caveat, which is now about freshness.

    Document *durability* across workers is solved: the corpus is a SQLite
    database and every mutation is a transaction touching its own rows, so
    concurrent uploads no longer overwrite each other. That was the serious
    bug — a worker would acknowledge an upload with a 200 and another worker's
    whole-corpus rewrite would silently erase it.

    What remains is weaker and bounded: each worker keeps its **search index**
    in memory, built at startup. A document uploaded through worker 1 is safely
    on disk immediately, but worker 2 will not retrieve it until worker 2
    restarts. So the same question can be answered or refused depending on
    which worker handles it.

    This is a staleness window, not data loss, and it closes at the next
    restart. Fixing it properly means workers detecting each other's writes and
    rebuilding — a different piece of work, deliberately not bundled with the
    durability fix.
    """
    workers = os.environ.get("WEB_CONCURRENCY") or os.environ.get("UVICORN_WORKERS")
    try:
        count = int(workers) if workers else 1
    except ValueError:
        count = 1

    if count > 1:
        logger.warning(
            "Running %s workers over DATA_DIR=%s. Uploads are durable and "
            "transactional, but each worker's search index is built at startup: "
            "a document uploaded via one worker is not retrievable from another "
            "until that worker restarts.",
            count,
            settings.data_dir,
        )


_warn_if_multiprocess()


def current_tenant(
    x_api_key: Optional[str] = Header(
        default=None,
        alias="X-API-Key",
        description=(
            "Tenant credential. Required when the service runs in multi-tenant "
            "mode; ignored otherwise."
        ),
    ),
) -> TenantContext:
    """Resolve the caller's tenant from its credential.

    This is the only way an endpoint obtains a store, which is what makes the
    isolation structural rather than a convention: an endpoint cannot reach a
    corpus without first going through authentication, because there is no
    module-level store left to reach for.

    Note that the tenant is never read from the request body or path. A caller
    may say what it wants done, never whose data to do it to.
    """
    if not settings.multi_tenant:
        return registry.context_for(DEFAULT_TENANT)

    try:
        tenant = registry.resolve(x_api_key)
    except UnknownTenantError as exc:
        # 401, not 403: the credential is missing or unrecognised, so the
        # correct remedy is to authenticate, not to request permission.
        raise HTTPException(status_code=401, detail=str(exc))
    return registry.context_for(tenant)

app = FastAPI(
    title="Hybrid Scoped RAG",
    description=(
        "Retrieval-augmented chat with two guarantees: **hybrid retrieval** "
        "(BM25 + dense vectors fused by Reciprocal Rank Fusion) so questions "
        "match by wording *and* by meaning, and **scoped retrieval** (metadata "
        "filters applied before scoring) so the bot can only ever see the slice "
        "of the corpus you allow."
    ),
    version="2.0.0",
)


@app.exception_handler(LLMUnavailableError)
async def _llm_unavailable_handler(_request, exc: LLMUnavailableError):
    # 503: the service is fine, its upstream dependency is not.
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(InvalidFilterError)
async def _invalid_filter_handler(_request, exc: InvalidFilterError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(InvalidTenantError)
async def _invalid_tenant_handler(_request, exc: InvalidTenantError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _parse_scope(
    raw_scope, doc_ids: Optional[List[str]]
) -> Optional[ScopeFilter]:
    """Build a ScopeFilter, turning a bad filter into a 400 rather than a 500."""
    if not raw_scope and not doc_ids:
        return None
    try:
        return ScopeFilter.parse(raw_scope, doc_ids)
    except InvalidFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------


@app.get("/api/status", response_model=StatusResponse, tags=["status"])
def status(ctx: TenantContext = Depends(current_tenant)) -> StatusResponse:
    """Health and configuration summary.

    In multi-tenant mode this still requires a key: document and chunk counts
    describe a specific corpus, and serving them unauthenticated would disclose
    one tenant's holdings to anyone who asked.
    """
    store, personas = ctx.store, ctx.personas
    persona = personas.current
    return StatusResponse(
        status="ok",
        llm_configured=settings.has_api_key,
        model=settings.model,
        documents=len(store.list_documents()),
        chunks=store.total_chunks,
        persona=PersonaModel(**persona.to_dict()),
        greeting=persona.effective_greeting(),
        strict_scope=settings.strict_scope,
        relevance_floor=settings.relevance_floor,
        retrieval=RetrievalInfo(
            embedding_backend=embedder.name,
            embedding_dimensions=embedder.dimensions,
            rrf_k=settings.rrf_k,
            lexical_weight=settings.lexical_weight,
            dense_weight=settings.dense_weight,
            dense_rescue_floor=settings.dense_rescue_floor,
            hybrid_enabled=settings.hybrid_enabled,
        ),
        facets=store.facets(),
        multi_tenant=settings.multi_tenant,
        tenant_id=ctx.tenant_id,
        tenant_name=ctx.tenant.display_name,
    )


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


@app.post("/api/documents", response_model=UploadResponse, tags=["documents"])
async def upload_documents(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(
        default=None,
        description=(
            'JSON object applied as metadata to every file in this batch, e.g. '
            '{"course": "CS1428", "term": "fall2025"}.'
        ),
    ),
    ctx: TenantContext = Depends(current_tenant),
) -> UploadResponse:
    """Ingest one or more documents into the knowledge base.

    Partial success is intentional: one bad file in a batch should not discard
    the good ones. Failures are reported per-file in ``skipped``.

    ``metadata`` arrives as a JSON string because this is a multipart form
    upload, which has no native nested-object encoding.
    """
    store = ctx.store
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    parsed_metadata = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="metadata must be a valid JSON object: {}".format(exc),
            )
        if not isinstance(parsed_metadata, dict):
            raise HTTPException(
                status_code=400, detail="metadata must be a JSON object."
            )

    uploaded, skipped = [], []

    for upload in files:
        filename = upload.filename or "unnamed"
        try:
            data = await upload.read()
        except Exception as exc:
            skipped.append(
                {"name": filename, "reason": "Could not read: {}".format(exc)}
            )
            continue
        finally:
            await upload.close()

        if len(data) > settings.max_upload_bytes:
            skipped.append(
                {
                    "name": filename,
                    "reason": "File is {:.1f} MB; the limit is {} MB.".format(
                        len(data) / 1024 / 1024, settings.max_upload_mb
                    ),
                }
            )
            continue

        try:
            doc = store.add(filename, data, metadata=parsed_metadata)
            uploaded.append(DocumentModel(**doc.to_dict()))
        except DuplicateDocumentError as exc:
            skipped.append({"name": filename, "reason": str(exc)})
        except ExtractionError as exc:
            skipped.append({"name": filename, "reason": str(exc)})
        except InvalidFilterError as exc:
            skipped.append({"name": filename, "reason": str(exc)})
        except Exception as exc:
            logger.exception("Failed to ingest %s", filename)
            skipped.append(
                {"name": filename, "reason": "Unexpected error: {}".format(exc)}
            )

    if not uploaded and skipped:
        # Nothing landed — surface it as an error rather than a misleading 200.
        raise HTTPException(
            status_code=400,
            detail="No files could be ingested. "
            + "; ".join("{}: {}".format(s["name"], s["reason"]) for s in skipped),
        )

    return UploadResponse(
        uploaded=uploaded,
        skipped=skipped,
        total_documents=len(store.list_documents()),
        total_chunks=store.total_chunks,
    )


@app.get("/api/documents", response_model=List[DocumentModel], tags=["documents"])
def list_documents(
    ctx: TenantContext = Depends(current_tenant),
) -> List[DocumentModel]:
    """List everything currently in the caller's knowledge base."""
    return [DocumentModel(**d.to_dict()) for d in ctx.store.list_documents()]


@app.put(
    "/api/documents/{doc_id}/metadata",
    response_model=DocumentModel,
    tags=["documents"],
)
def set_document_metadata(
    doc_id: str,
    body: MetadataRequest,
    ctx: TenantContext = Depends(current_tenant),
) -> DocumentModel:
    """Replace a document's metadata tags, re-scoping it without re-uploading.

    ``doc_id`` is caller-supplied, so this is where a horizontal-privilege bug
    would live. It cannot: the lookup runs against the caller's own store, so
    another tenant's id is simply absent and returns 404 — which is also the
    right answer to disclose, since confirming the id exists elsewhere would
    leak that the document exists at all.
    """
    try:
        doc = ctx.store.set_metadata(doc_id, body.metadata)
    except InvalidFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if doc is None:
        raise HTTPException(
            status_code=404, detail="No document with id '{}'.".format(doc_id)
        )
    return DocumentModel(**doc.to_dict())


@app.get("/api/facets", response_model=FacetsResponse, tags=["documents"])
def facets(ctx: TenantContext = Depends(current_tenant)) -> FacetsResponse:
    """Every metadata key and value in the caller's corpus, with doc counts.

    The UI uses this to offer real scope choices rather than making the user
    guess at tag names.
    """
    return FacetsResponse(facets=ctx.store.facets())


@app.delete("/api/documents/{doc_id}", tags=["documents"])
def delete_document(doc_id: str, ctx: TenantContext = Depends(current_tenant)):
    """Remove one of the caller's own documents and de-index its chunks."""
    store = ctx.store
    if not store.delete(doc_id):
        raise HTTPException(
            status_code=404, detail="No document with id '{}'.".format(doc_id)
        )
    return {
        "deleted": doc_id,
        "total_documents": len(store.list_documents()),
        "total_chunks": store.total_chunks,
    }


@app.delete("/api/documents", tags=["documents"])
def clear_documents(ctx: TenantContext = Depends(current_tenant)):
    """Wipe the caller's knowledge base. Other tenants are untouched."""
    ctx.store.clear()
    return {"cleared": True, "total_documents": 0, "total_chunks": 0}


@app.get("/api/supported-types", tags=["documents"])
def supported_types():
    return {"extensions": sorted(SUPPORTED_EXTENSIONS)}


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


@app.post("/api/search", response_model=SearchResponse, tags=["retrieval"])
def search(
    body: SearchRequest, ctx: TenantContext = Depends(current_tenant)
) -> SearchResponse:
    """Run hybrid retrieval and return the ranked passages. No LLM call.

    This endpoint exists so the retrieval system can be inspected and graded on
    its own terms, with no API key and no generation cost: it exposes the
    per-retriever ranks that RRF fused, and the decision the scope gate would
    have reached.
    """
    store = ctx.store
    scope = _parse_scope(body.scope, body.doc_ids)
    matched = store.count_matching(scope)

    hits = store.index.search(body.query, top_k=body.top_k, scope=scope)
    decision = classify(
        question=body.query,
        hits=hits,
        relevance_floor=settings.relevance_floor,
        corpus_empty=store.is_empty,
        scope=scope,
        scope_matched_documents=matched,
        dense_rescue_floor=settings.dense_rescue_floor,
    )

    return SearchResponse(
        query=body.query,
        hits=[
            SearchHitModel(
                doc_name=h.chunk.doc_name,
                doc_id=h.chunk.doc_id,
                chunk_id=h.chunk.chunk_id,
                position=h.chunk.position,
                text=h.chunk.text,
                fused_score=round(h.score, 6),
                lexical_score=round(h.normalized, 4),
                dense_score=round(h.dense_score, 4),
                lexical_rank=h.lexical_rank,
                dense_rank=h.dense_rank,
                retrievers=h.retrievers,
                metadata=dict(h.chunk.metadata),
            )
            for h in hits
        ],
        coverage=round(hits[0].coverage, 4) if hits else 0.0,
        documents_in_scope=matched,
        scope_description=scope.describe() if scope and not scope.is_empty else "",
        decision=decision.decision,
        reason=decision.reason,
    )


# --------------------------------------------------------------------------
# Persona
# --------------------------------------------------------------------------


@app.get("/api/persona", response_model=PersonaModel, tags=["persona"])
def get_persona(ctx: TenantContext = Depends(current_tenant)) -> PersonaModel:
    return PersonaModel(**ctx.personas.current.to_dict())


@app.put("/api/persona", response_model=PersonaModel, tags=["persona"])
def set_persona(
    body: PersonaModel, ctx: TenantContext = Depends(current_tenant)
) -> PersonaModel:
    """Define who the bot is and what subject it is allowed to cover.

    Per tenant: each customer brands and scopes its own assistant, and one
    tenant's persona change must not alter another's bot.
    """
    saved = ctx.personas.set(Persona.from_dict(body.model_dump()))
    return PersonaModel(**saved.to_dict())


@app.post("/api/persona/preset", response_model=PersonaModel, tags=["persona"])
def apply_preset(
    body: PresetRequest, ctx: TenantContext = Depends(current_tenant)
) -> PersonaModel:
    """Switch to a built-in persona (customer_care, course_ta, generic)."""
    try:
        saved = ctx.personas.apply_preset(body.preset)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return PersonaModel(**saved.to_dict())


@app.get("/api/persona/presets", tags=["persona"])
def list_presets():
    return {key: value.to_dict() for key, value in PRESETS.items()}


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


@app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
def chat(
    body: ChatRequest, ctx: TenantContext = Depends(current_tenant)
) -> ChatResponse:
    """Ask a question. Off-topic questions are refused without an LLM call.

    The engine is shared but stateless; the corpus and persona it works from
    are passed in per request, so two tenants chatting concurrently cannot see
    each other's documents.
    """
    scope = _parse_scope(body.scope, body.doc_ids)
    result = engine.answer(
        question=body.message,
        persona=ctx.personas.current,
        history=[turn.model_dump() for turn in body.history],
        scope=scope,
        store=ctx.store,
    )
    return ChatResponse(**result.to_dict())


# --------------------------------------------------------------------------
# Web UI
# --------------------------------------------------------------------------

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(WEB_DIR / "index.html"))
