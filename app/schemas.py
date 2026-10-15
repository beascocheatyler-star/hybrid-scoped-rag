"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    # Constrained so a bad role is a 422 the client can see, rather than a turn
    # silently dropped from the history.
    role: Literal["user", "assistant"] = Field(
        ..., description="Either 'user' or 'assistant'."
    )
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's question.")
    history: List[ChatTurn] = Field(
        default_factory=list, description="Prior turns, oldest first."
    )
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Restrict retrieval to these documents. Omit to search all.",
    )
    scope: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Metadata filter applied before retrieval. AND across keys, OR "
            'within a key. Examples: {"course": "CS1428"}, '
            '{"course": ["CS1428", "CS2308"], "term": "fall2025"}, '
            '{"course": {"not": "CS1428"}}.'
        ),
    )


class CitationModel(BaseModel):
    doc_name: str
    doc_id: str
    chunk_id: str
    position: int
    score: float
    excerpt: str
    retrievers: List[str] = Field(
        default_factory=list,
        description="Which retrievers found this passage: lexical, dense, or both.",
    )
    lexical_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    in_scope: bool
    decision: str = Field(
        ...,
        description=(
            "in_scope | borderline | out_of_scope | small_talk | meta | "
            "no_corpus | empty_scope"
        ),
    )
    citations: List[CitationModel] = Field(default_factory=list)
    retrieval_score: float
    dense_score: float = 0.0
    rescued_by_dense: bool = False
    scope_description: str = ""
    documents_in_scope: int = 0
    reason: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None


class SearchRequest(BaseModel):
    """Retrieval only — no LLM call, so it works with no API key."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=50)
    doc_ids: Optional[List[str]] = None
    scope: Optional[Dict[str, Any]] = None


class SearchHitModel(BaseModel):
    doc_name: str
    doc_id: str
    chunk_id: str
    position: int
    text: str
    fused_score: float
    lexical_score: float
    dense_score: float
    lexical_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    retrievers: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    hits: List[SearchHitModel]
    coverage: float
    documents_in_scope: int
    scope_description: str
    decision: str = Field(
        ..., description="What the scope gate would decide for this query."
    )
    reason: str


class DocumentModel(BaseModel):
    doc_id: str
    name: str
    size_bytes: int
    chunk_count: int
    char_count: int
    sha256: str
    uploaded_at: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class MetadataRequest(BaseModel):
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Replaces the document's metadata entirely.",
    )


class UploadResponse(BaseModel):
    uploaded: List[DocumentModel]
    skipped: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Files that could not be ingested, with the reason why.",
    )
    total_documents: int
    total_chunks: int


class FacetsResponse(BaseModel):
    facets: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Metadata key -> value -> number of documents carrying it.",
    )


class PersonaModel(BaseModel):
    name: str = "Assistant"
    topic: str = "the uploaded documents"
    style: str = "Be clear, concise, and professional."
    greeting: str = ""
    suggested_questions: List[str] = Field(default_factory=list)


class PresetRequest(BaseModel):
    preset: str = Field(..., description="customer_care | course_ta | generic")


class RetrievalInfo(BaseModel):
    embedding_backend: str
    embedding_dimensions: int
    rrf_k: int
    lexical_weight: float
    dense_weight: float
    dense_rescue_floor: float
    hybrid_enabled: bool


class StatusResponse(BaseModel):
    status: str
    llm_configured: bool
    model: str
    documents: int
    chunks: int
    persona: PersonaModel
    greeting: str
    strict_scope: bool
    relevance_floor: float
    retrieval: RetrievalInfo
    facets: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str
