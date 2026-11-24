"""/api/v2: notebooks, sources, notebook-scoped search, and jobs.

Every route resolves the notebook through the caller's membership, so a
notebook the user cannot see answers 404, the same as one that does not
exist. In SINGLE_USER_MODE (the default) the caller is always the built-in
``local`` user, who owns every notebook created without a login.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import Settings
from app.core.extract import SUPPORTED_EXTENSIONS
from app.core.fetch import FetchError
from app.core.guard import classify
from app.core.metadata import InvalidFilterError, ScopeFilter
from app.core.notebooks import (
    DEFAULT_NOTEBOOK_ID,
    LOCAL_USER_ID,
    NotebookNotFound,
    NotebookStore,
    SourceLimitError,
)
from app.core.services import Services
from app.core.store import DuplicateDocumentError
from app.schemas import SearchHitModel, SearchResponse

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class NotebookIn(BaseModel):
    title: str = Field(default="Untitled notebook", max_length=200)
    description: str = Field(default="", max_length=2000)
    emoji: str = Field(default="", max_length=16)


class NotebookPatch(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    emoji: Optional[str] = Field(default=None, max_length=16)


class NotebookOut(BaseModel):
    id: str
    owner_id: str
    title: str
    description: str
    emoji: str
    created_at: str
    updated_at: str
    source_count: int = 0
    is_default: bool = False


class CopyIn(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)


class SourceOut(BaseModel):
    id: str
    notebook_id: str
    kind: str
    title: str
    original_name: str
    mime: str
    sha256: str
    size_bytes: int
    uri: str
    status: str
    error: str
    metadata: Dict[str, str]
    char_count: int
    chunk_count: int
    created_at: str
    updated_at: str
    has_original: bool
    guide: Dict[str, Any] = Field(default_factory=dict)


class SourcePatch(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    metadata: Optional[Dict[str, Any]] = None


class TextSourceIn(BaseModel):
    kind: str = Field(description="'text', 'url' or 'youtube'.")
    title: str = Field(default="", max_length=200)
    text: str = Field(default="", max_length=5_000_000)
    url: str = Field(default="", max_length=4000)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueuedSource(BaseModel):
    job_id: str
    source_id: str
    name: str


class SkippedSource(BaseModel):
    name: str
    reason: str


class AddSourcesOut(BaseModel):
    queued: List[QueuedSource]
    skipped: List[SkippedSource]


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    source_ids: Optional[List[str]] = None
    scope: Optional[Dict[str, Any]] = None
    top_k: int = Field(default=6, ge=1, le=50)


class JobOut(BaseModel):
    id: str
    notebook_id: Optional[str]
    type: str
    status: str
    progress: float
    result: Dict[str, Any]
    error: str
    started_at: Optional[str]
    finished_at: Optional[str]
    created_at: str
    updated_at: str


class MeOut(BaseModel):
    id: str
    display_name: str
    single_user_mode: bool


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------


def create_v2_router(services: Services, settings: Settings) -> APIRouter:
    r = APIRouter(prefix="/api/v2", tags=["v2"])
    registry = services.registry
    jobs = services.jobs
    db = services.db

    def current_user() -> str:
        # M8 replaces this with session auth when SINGLE_USER_MODE=false.
        return LOCAL_USER_ID

    def can_access(nb_id: str, user_id: str) -> bool:
        return db.one(
            "SELECT 1 FROM memberships WHERE notebook_id = ? AND user_id = ?",
            (nb_id, user_id),
        ) is not None

    def notebook_store(nb_id: str) -> NotebookStore:
        if not can_access(nb_id, current_user()):
            raise HTTPException(status_code=404, detail="Notebook not found.")
        try:
            return registry.store(nb_id)
        except NotebookNotFound:
            raise HTTPException(status_code=404, detail="Notebook not found.")

    def nb_out(nb) -> NotebookOut:
        n = db.one("SELECT COUNT(*) AS n FROM sources WHERE notebook_id = ?", (nb.id,))
        return NotebookOut(
            id=nb.id, owner_id=nb.owner_id, title=nb.title, description=nb.description,
            emoji=nb.emoji, created_at=nb.created_at, updated_at=nb.updated_at,
            source_count=int(n["n"]), is_default=nb.id == DEFAULT_NOTEBOOK_ID,
        )

    def parse_scope(raw, ids) -> Optional[ScopeFilter]:
        if not raw and not ids:
            return None
        try:
            return ScopeFilter.parse(raw, ids)
        except InvalidFilterError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # --- me --------------------------------------------------------------

    @r.get("/me", response_model=MeOut)
    def me() -> MeOut:
        row = db.one("SELECT id, display_name FROM users WHERE id = ?", (current_user(),))
        return MeOut(id=row["id"], display_name=row["display_name"],
                     single_user_mode=settings.single_user_mode)

    # --- notebooks -------------------------------------------------------

    @r.get("/notebooks", response_model=List[NotebookOut])
    def list_notebooks() -> List[NotebookOut]:
        return [nb_out(nb) for nb in registry.list(current_user())]

    @r.post("/notebooks", response_model=NotebookOut, status_code=201)
    def create_notebook(body: NotebookIn) -> NotebookOut:
        return nb_out(registry.create(body.title, body.description, body.emoji,
                                      owner_id=current_user()))

    @r.get("/notebooks/{nb_id}", response_model=NotebookOut)
    def get_notebook(nb_id: str) -> NotebookOut:
        notebook_store(nb_id)
        return nb_out(registry.get(nb_id))

    @r.patch("/notebooks/{nb_id}", response_model=NotebookOut)
    def patch_notebook(nb_id: str, body: NotebookPatch) -> NotebookOut:
        notebook_store(nb_id)
        try:
            return nb_out(registry.update(nb_id, body.title, body.description, body.emoji))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @r.delete("/notebooks/{nb_id}", status_code=204)
    def delete_notebook(nb_id: str) -> Response:
        notebook_store(nb_id)
        try:
            registry.delete(nb_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return Response(status_code=204)

    @r.post("/notebooks/{nb_id}/copy", response_model=NotebookOut, status_code=201)
    def copy_notebook(nb_id: str, body: Optional[CopyIn] = None) -> NotebookOut:
        notebook_store(nb_id)
        return nb_out(registry.copy(nb_id, body.title if body else None,
                                    owner_id=current_user()))

    # --- sources ---------------------------------------------------------

    @r.post("/notebooks/{nb_id}/sources", response_model=AddSourcesOut, status_code=202)
    async def add_sources(
        nb_id: str,
        request: Request,
        wait: bool = Query(False, description="Block until ingestion finishes."),
    ) -> AddSourcesOut:
        """Add sources. Multipart ``files`` (+ optional ``metadata`` JSON form
        field), or a JSON body ``{"kind": "text", "title", "text", "metadata"}``.
        Each source is ingested by a background job; poll ``/jobs/{id}``."""
        store = notebook_store(nb_id)
        items = []  # (name, data, metadata, kind, mime)
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("application/json"):
            try:
                body = TextSourceIn(**(await request.json()))
            except Exception as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            if body.kind in ("url", "youtube"):
                if not body.url.strip():
                    raise HTTPException(status_code=400, detail="url must not be empty.")
                try:
                    src = store.create_remote_source(body.kind, body.url, body.title, body.metadata)
                except FetchError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
                except (DuplicateDocumentError, SourceLimitError, InvalidFilterError, ValueError) as exc:
                    raise HTTPException(status_code=409 if isinstance(exc, DuplicateDocumentError) else 400,
                                        detail=str(exc))
                job = jobs.enqueue("ingest", {"source_id": src.id}, notebook_id=nb_id)
                if wait:
                    jobs.wait(job.id, timeout=300)
                return AddSourcesOut(queued=[QueuedSource(job_id=job.id, source_id=src.id, name=src.title)],
                                     skipped=[])
            if body.kind != "text":
                raise HTTPException(status_code=400,
                                    detail="Unknown source kind {!r}.".format(body.kind))
            if not body.text.strip():
                raise HTTPException(status_code=400, detail="text must not be empty.")
            title = body.title or "Pasted text"
            name = Path(title).name or "Pasted text"
            items.append((name + ".txt" if not name.endswith(".txt") else name,
                          body.text.encode("utf-8"), body.metadata, "text", "text/plain",
                          title))
        elif ctype.startswith("multipart/form-data"):
            form = await request.form()
            meta: Dict[str, Any] = {}
            raw_meta = form.get("metadata")
            if raw_meta:
                try:
                    meta = json.loads(str(raw_meta))
                except ValueError as exc:
                    raise HTTPException(status_code=400,
                                        detail="metadata must be a valid JSON object: {}".format(exc))
                if not isinstance(meta, dict):
                    raise HTTPException(status_code=400, detail="metadata must be a JSON object.")
            for up in form.getlist("files"):
                if not hasattr(up, "read"):
                    continue
                data = await up.read()
                items.append((up.filename or "unnamed", data, meta, "file",
                              getattr(up, "content_type", "") or "", None))
        else:
            raise HTTPException(status_code=415,
                                detail="Send multipart/form-data files or a JSON text source.")
        if not items:
            raise HTTPException(status_code=400, detail="No sources were provided.")

        queued: List[QueuedSource] = []
        skipped: List[SkippedSource] = []
        for name, data, meta, kind, mime, title in items:
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                skipped.append(SkippedSource(name=name, reason="Unsupported file type {!r}.".format(ext or name)))
                continue
            if len(data) > settings.max_upload_bytes:
                skipped.append(SkippedSource(name=name, reason="File is {:.1f} MB; the limit is {} MB.".format(
                    len(data) / 1024 / 1024, settings.max_upload_mb)))
                continue
            try:
                src = store.create_source(name, data, meta, kind=kind, mime=mime, title=title)
            except (DuplicateDocumentError, SourceLimitError, InvalidFilterError, ValueError) as exc:
                skipped.append(SkippedSource(name=name, reason=str(exc)))
                continue
            job = jobs.enqueue("ingest", {"source_id": src.id}, notebook_id=nb_id)
            queued.append(QueuedSource(job_id=job.id, source_id=src.id, name=src.title))

        if wait:
            for q in queued:
                jobs.wait(q.job_id, timeout=300)
        if not queued and skipped:
            raise HTTPException(status_code=400, detail="No sources could be added. " + "; ".join(
                "{}: {}".format(s.name, s.reason) for s in skipped))
        return AddSourcesOut(queued=queued, skipped=skipped)

    @r.get("/notebooks/{nb_id}/sources", response_model=List[SourceOut])
    def list_sources(nb_id: str) -> List[SourceOut]:
        return [SourceOut(**s.to_dict()) for s in notebook_store(nb_id).list_sources()]

    def get_src(store: NotebookStore, source_id: str):
        src = store.get_source(source_id)
        if src is None:
            raise HTTPException(status_code=404, detail="Source not found.")
        return src

    @r.get("/notebooks/{nb_id}/sources/{source_id}", response_model=SourceOut)
    def get_source(nb_id: str, source_id: str) -> SourceOut:
        return SourceOut(**get_src(notebook_store(nb_id), source_id).to_dict())

    @r.get("/notebooks/{nb_id}/sources/{source_id}/content")
    def source_content(nb_id: str, source_id: str,
                       highlight: Optional[str] = Query(None, description="chunk id to highlight")
                       ) -> Dict[str, Any]:
        """Extracted text plus chunk anchors. With ``highlight``, also returns
        that chunk's span so the viewer can scroll to and mark it."""
        store = notebook_store(nb_id)
        src = get_src(store, source_id)
        if src.status != "ready":
            raise HTTPException(status_code=409, detail="Source is {}.".format(src.status))
        chunks = store.source_chunks(src.id)
        out: Dict[str, Any] = {
            "source_id": src.id, "title": src.title, "kind": src.kind, "uri": src.uri,
            "text": store.extracted_text(src.id) or "",
            "anchors": [{k: c[k] for k in ("id", "position", "start_char", "end_char", "page",
                                            "slide", "t_start_ms", "t_end_ms", "section_path")}
                        for c in chunks],
            "highlight": None,
        }
        if highlight:
            hit = next((a for a in out["anchors"] if a["id"] == highlight), None)
            if hit is None:
                raise HTTPException(status_code=404, detail="Chunk not found in this source.")
            out["highlight"] = hit
        return out

    @r.get("/notebooks/{nb_id}/sources/{source_id}/file")
    def source_file(nb_id: str, source_id: str):
        store = notebook_store(nb_id)
        src = get_src(store, source_id)
        path = store.original_path(src.id)
        if path is None:
            raise HTTPException(status_code=404, detail="No original file is stored for this source.")
        return FileResponse(str(path), filename=src.original_name,
                            media_type=src.mime or "application/octet-stream")

    @r.patch("/notebooks/{nb_id}/sources/{source_id}", response_model=SourceOut)
    def patch_source(nb_id: str, source_id: str, body: SourcePatch) -> SourceOut:
        store = notebook_store(nb_id)
        get_src(store, source_id)
        try:
            if body.title is not None:
                store.rename(source_id, body.title)
            if body.metadata is not None:
                store.set_metadata(source_id, body.metadata)
        except (InvalidFilterError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return SourceOut(**get_src(store, source_id).to_dict())

    @r.delete("/notebooks/{nb_id}/sources/{source_id}", status_code=204)
    def delete_source(nb_id: str, source_id: str) -> Response:
        store = notebook_store(nb_id)
        if not store.delete(source_id):
            raise HTTPException(status_code=404, detail="Source not found.")
        return Response(status_code=204)

    @r.get("/notebooks/{nb_id}/facets")
    def facets(nb_id: str) -> Dict[str, Dict[str, int]]:
        return notebook_store(nb_id).facets()

    # --- search (no LLM, no key: invariant I9) ---------------------------

    @r.post("/notebooks/{nb_id}/search", response_model=SearchResponse)
    def search(nb_id: str, body: SearchIn) -> SearchResponse:
        store = notebook_store(nb_id)
        scope = parse_scope(body.scope, body.source_ids)
        matched = store.count_matching(scope)
        hits = store.index.search(body.query, top_k=body.top_k, scope=scope)
        decision = classify(
            question=body.query, hits=hits, relevance_floor=settings.relevance_floor,
            corpus_empty=store.is_empty, scope=scope, scope_matched_documents=matched,
            dense_rescue_floor=settings.dense_rescue_floor,
        )
        return SearchResponse(
            query=body.query,
            hits=[
                SearchHitModel(
                    doc_name=h.chunk.doc_name, doc_id=h.chunk.doc_id,
                    chunk_id=h.chunk.chunk_id, position=h.chunk.position, text=h.chunk.text,
                    fused_score=round(h.score, 6), lexical_score=round(h.normalized, 4),
                    dense_score=round(h.dense_score, 4), lexical_rank=h.lexical_rank,
                    dense_rank=h.dense_rank, retrievers=h.retrievers,
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

    # --- jobs ------------------------------------------------------------

    def job_for_user(job_id: str):
        job = jobs.get(job_id)
        if job is None or (job.notebook_id and not can_access(job.notebook_id, current_user())):
            raise HTTPException(status_code=404, detail="Job not found.")
        return job

    @r.get("/jobs/{job_id}", response_model=JobOut)
    def get_job(job_id: str) -> JobOut:
        d = job_for_user(job_id).to_dict()
        d.pop("payload", None)
        return JobOut(**d)

    @r.post("/jobs/{job_id}/cancel", response_model=JobOut)
    def cancel_job(job_id: str) -> JobOut:
        job_for_user(job_id)
        d = jobs.cancel(job_id).to_dict()  # type: ignore[union-attr]
        d.pop("payload", None)
        return JobOut(**d)

    return r
