"""Runtime configuration, loaded from environment with sane defaults.

Everything a reviewer might want to change is an env var with a working default,
so the app boots with zero configuration — and, unlike the predecessor, runs
usefully with no API key at all: retrieval, scoping, and the scope gate are
entirely local. Only the final answer-generation step needs a key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    # --- LLM ---
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    model: str = field(default_factory=lambda: os.getenv("MODEL", "claude-opus-5-5"))
    # Per-task models (M2+); each defaults to MODEL when unset.
    model_chat: str = field(default_factory=lambda: os.getenv("MODEL_CHAT", ""))
    model_rewrite: str = field(default_factory=lambda: os.getenv("MODEL_REWRITE", ""))
    model_studio: str = field(default_factory=lambda: os.getenv("MODEL_STUDIO", ""))
    model_judge: str = field(default_factory=lambda: os.getenv("MODEL_JUDGE", ""))
    # anthropic (default) | openai  (any OpenAI-compatible server: Ollama, vLLM, ...)
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic").lower())
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    # Server-side refusal fallback on Anthropic (re-runs a declined request on
    # a fallback model in the same call). LLM_REFUSAL_FALLBACK=false disables.
    refusal_fallback: bool = field(default_factory=lambda: _env_bool("LLM_REFUSAL_FALLBACK", True))
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 2000))
    effort: str = field(default_factory=lambda: os.getenv("EFFORT", "medium"))

    # --- Chunking ---
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1200))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 200))

    # --- Retrieval ---
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 6))
    # Below this combined relevance score, the question is treated as out of
    # scope before we ever spend a token on the LLM. See app/core/guard.py for
    # how the score is computed and docs/scope-tuning.md for how to tune it.
    relevance_floor: float = field(
        default_factory=lambda: _env_float("RELEVANCE_FLOOR", 0.42)
    )

    # --- Hybrid fusion ---
    # Which embedder backs dense retrieval: "hashing" (default, zero
    # dependencies, zero download) or "sentence-transformers" (real semantic
    # vectors, optional install). See app/core/embedding.py.
    embedding_backend: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "auto")
    )
    embedding_dimensions: int = field(
        default_factory=lambda: _env_int("EMBEDDING_DIMENSIONS", 1024)
    )
    # RRF smoothing constant. See app/core/retrieval.py for what it controls.
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    # Relative influence of each retriever in the fusion. Setting either to 0
    # disables that half, which is how scripts/measure_hybrid.py isolates the
    # contribution of each.
    lexical_weight: float = field(
        default_factory=lambda: _env_float("LEXICAL_WEIGHT", 1.0)
    )
    dense_weight: float = field(
        default_factory=lambda: _env_float("DENSE_WEIGHT", 1.0)
    )
    # Dense similarity at or above which a question failing the lexical gate is
    # promoted to borderline rather than refused. Set to 1.0 to disable.
    dense_rescue_floor: float = field(
        default_factory=lambda: _env_float("DENSE_RESCUE_FLOOR", 0.36)
    )

    # --- Query understanding (M3) ---
    # heuristic (default): condense follow-ups with the previous question;
    # llm: use MODEL_REWRITE for that; off: retrieve on the raw turn only.
    query_rewrite: str = field(default_factory=lambda: os.getenv("QUERY_REWRITE", "heuristic").lower())
    multi_query: bool = field(default_factory=lambda: _env_bool("MULTI_QUERY", False))
    reranker: str = field(default_factory=lambda: os.getenv("RERANKER", "off").lower())
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    # Pass every in-scope chunk instead of top-k when they fit this many
    # tokens. 0 = off (default here, because it multiplies prompt cost; the
    # roadmap suggests 150000 for quality-first deployments).
    long_context_token_budget: int = field(
        default_factory=lambda: _env_int("LONG_CONTEXT_TOKEN_BUDGET", 0))

    # --- Storage ---
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", str(REPO_ROOT / "data")))
    )

    # --- Notebooks / jobs (M1) ---
    # Every request acts as the built-in "local" user; no login. Multi-user
    # auth arrives in M8 and is opt-in by setting this to false.
    single_user_mode: bool = field(
        default_factory=lambda: _env_bool("SINGLE_USER_MODE", True)
    )
    job_workers: int = field(default_factory=lambda: _env_int("JOB_WORKERS", 2))
    # Generate a Source Guide (summary, topics, questions) after each ingest.
    source_guides: bool = field(default_factory=lambda: _env_bool("SOURCE_GUIDES", True))
    max_sources_per_notebook: int = field(
        default_factory=lambda: _env_int("MAX_SOURCES_PER_NOTEBOOK", 50)
    )

    # --- Upload limits ---
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 200))
    # Per-source cap on extracted words (NotebookLM uses 500k).
    max_source_words: int = field(
        default_factory=lambda: _env_int("MAX_SOURCE_WORDS", 500_000)
    )

    # --- Behaviour ---
    # When True, the bot refuses anything not grounded in the uploaded corpus.
    strict_scope: bool = field(default_factory=lambda: _env_bool("STRICT_SCOPE", True))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "hsr.sqlite3"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key.strip())

    @property
    def hybrid_enabled(self) -> bool:
        return self.lexical_weight > 0.0 and self.dense_weight > 0.0

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "uploads").mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings


def reset_settings() -> None:
    """Drop the cached settings (used by tests that tweak the environment)."""
    global _settings
    _settings = None
