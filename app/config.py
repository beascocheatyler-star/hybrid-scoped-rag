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
    model: str = field(default_factory=lambda: os.getenv("MODEL", "claude-opus-5"))
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
        default_factory=lambda: _env_float("RELEVANCE_FLOOR", 0.40)
    )

    # --- Hybrid fusion ---
    # Which embedder backs dense retrieval: "hashing" (default, zero
    # dependencies, zero download) or "sentence-transformers" (real semantic
    # vectors, optional install). See app/core/embedding.py.
    embedding_backend: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "hashing")
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
        default_factory=lambda: _env_float("DENSE_RESCUE_FLOOR", 0.25)
    )

    # --- Storage ---
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", str(REPO_ROOT / "data")))
    )

    # --- Upload limits ---
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 25))

    # --- Behaviour ---
    # When True, the bot refuses anything not grounded in the uploaded corpus.
    strict_scope: bool = field(default_factory=lambda: _env_bool("STRICT_SCOPE", True))

    # --- Multi-tenancy ---
    # Off by default, which keeps every existing single-deployment install
    # working unchanged: with this False the service exposes one corpus and
    # needs no credentials, exactly as before.
    #
    # The permissive default is a deliberate compatibility choice and it has a
    # real failure mode — someone serving several customers who forgets to set
    # it gets no isolation while assuming they have it. Two mitigations, both in
    # main.py: the active mode is reported by /api/status, and a warning is
    # logged at startup when tenants are registered but the flag is off.
    multi_tenant: bool = field(default_factory=lambda: _env_bool("MULTI_TENANT", False))

    # Tenant credentials as "tenant_id:name:api_key", separated by commas or
    # newlines. Suitable for a demo or a small deployment; a real one should
    # back TenantRegistry with a database and rotate keys. See
    # docs/multi-tenancy.md.
    tenant_config: str = field(default_factory=lambda: os.getenv("TENANTS", ""))

    # The tenant used when multi-tenancy is off, so one code path serves both
    # modes rather than scattering `if multi_tenant` across every endpoint.
    default_tenant_id: str = field(
        default_factory=lambda: os.getenv("DEFAULT_TENANT_ID", "default")
    )

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
