"""Dense vector embeddings — the *hybrid* half of hybrid scoped RAG.

The predecessor was BM25 only, and documented why: a transformer embedder means
either an API key (cost, network) or a multi-hundred-megabyte download. Both are
real costs, and the scope gate runs on *every* question, so the embedder must be
cheap.

This module keeps that constraint and still adds a genuine semantic signal, by
making the embedder pluggable with a zero-dependency default:

``HashingEmbedder`` (default)
    Character n-gram hashing into a fixed-width L2-normalised vector. No
    dependencies, no download, deterministic, microseconds per chunk. It is
    *not* a semantic model — it will not learn that "exam" relates to
    "midterm". What it does provide is robustness to the failure mode BM25
    cannot handle at all: morphological and spelling variation. "enrolment"
    vs "enrollment", "returns" vs "return", "syllabus" vs "syllabi" share
    character n-grams and therefore cosine similarity, while BM25 sees
    unrelated tokens.

``SentenceTransformerEmbedder`` (optional)
    True semantic embeddings via ``sentence-transformers``, used automatically
    when the package is installed and ``EMBEDDING_BACKEND=sentence-transformers``
    is set. This is the upgrade path for a reviewer who wants real dense
    retrieval and is willing to accept the download.

Being explicit about this matters more than overselling it: the hashing default
buys lexical robustness, not semantic understanding. The *fusion* architecture
in ``retrieval.py`` is what makes the system hybrid, and it is unchanged when
you swap the backend — which is the point of the interface.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# 1024 dimensions: wide enough that hash collisions between distinct n-grams are
# rare at corpus sizes this project targets, narrow enough that a dot product
# over pure Python lists stays fast. Measured on the bundled corpora, scoring
# 1,000 chunks takes ~12 ms.
DEFAULT_DIMENSIONS = 1024

# Character n-gram width. 3 and 4 together capture both short morphological
# stems ("ret", "urn") and longer distinctive fragments ("retu", "turn"),
# which is what gives the variant-spelling robustness described above.
_NGRAM_SIZES = (3, 4)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*")


class Embedder:
    """Interface for anything that turns text into a fixed-width vector."""

    name = "base"
    dimensions = DEFAULT_DIMENSIONS

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors.

    Both embedders return L2-normalised vectors, so this is a plain dot product.
    The normalisation check is deliberately absent: it would cost a pass over
    every vector on every query to catch a bug that the unit tests already
    cover.
    """
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _l2_normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class HashingEmbedder(Embedder):
    """Deterministic character-n-gram hashing embedder. No dependencies.

    Uses the *signed* hashing trick: each n-gram hashes to a bucket index and,
    independently, to a sign. Without the sign, every collision adds energy to a
    bucket and unrelated documents drift toward mutual similarity; with it,
    collisions cancel in expectation and the similarity estimate stays unbiased.
    """

    name = "hashing"

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        # blake2b with a small digest is markedly faster than sha256 here and
        # the cryptographic properties are irrelevant — only uniformity matters.
        self._digest_size = 8

    def _bucket(self, token: str) -> tuple:
        digest = hashlib.blake2b(
            token.encode("utf-8"), digest_size=self._digest_size
        ).digest()
        value = int.from_bytes(digest, "big")
        index = value % self.dimensions
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        if not text:
            return vector

        words = _WORD_RE.findall(text.lower())
        if not words:
            return vector

        for word in words:
            # The whole word carries the strongest signal, so weight it above
            # its fragments; n-grams exist to survive variation, not to replace
            # exact matching.
            index, sign = self._bucket(word)
            vector[index] += sign * 2.0

            # Boundary markers make a prefix n-gram distinct from the same
            # letters mid-word, which preserves stem information.
            padded = "^{}$".format(word)
            for n in _NGRAM_SIZES:
                if len(padded) < n:
                    continue
                for i in range(len(padded) - n + 1):
                    gram = padded[i : i + n]
                    index, sign = self._bucket(gram)
                    vector[index] += sign

        # Sublinear damping, mirroring BM25's treatment of term frequency: a
        # word occurring twenty times should not dominate one occurring twice.
        damped = [
            math.copysign(math.log1p(abs(v)), v) if v else 0.0 for v in vector
        ]
        return _l2_normalize(damped)


class SentenceTransformerEmbedder(Embedder):  # pragma: no cover - optional path
    """Real semantic embeddings, when the optional dependency is present."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "EMBEDDING_BACKEND=sentence-transformers requires the optional "
                "dependency. Install it with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading sentence-transformers model %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())
        self.model_name = model_name
        self._lock = threading.Lock()

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        # The model is not documented as thread-safe and FastAPI serves from a
        # worker pool.
        with self._lock:
            vectors = self._model.encode(
                list(texts), normalize_embeddings=True, show_progress_bar=False
            )
        return [[float(v) for v in vec] for vec in vectors]


_EMBEDDER: Optional[Embedder] = None
_EMBEDDER_LOCK = threading.Lock()


def build_embedder(backend: str, dimensions: int = DEFAULT_DIMENSIONS) -> Embedder:
    """Construct an embedder, falling back to hashing if the optional one fails.

    A missing optional dependency must not prevent the app from starting: the
    whole system still works on the hashing backend, so degrading loudly is
    better than refusing to boot.
    """
    choice = (backend or "hashing").strip().lower()
    if choice in {"sentence-transformers", "sentence_transformers", "st"}:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            logger.warning(
                "Could not initialise the sentence-transformers backend; "
                "falling back to the built-in hashing embedder.",
                exc_info=True,
            )
            return HashingEmbedder(dimensions)
    if choice not in {"hashing", "hash", "default"}:
        logger.warning(
            "Unknown EMBEDDING_BACKEND '%s'; using the hashing embedder.", backend
        )
    return HashingEmbedder(dimensions)


def get_embedder(
    backend: str = "hashing", dimensions: int = DEFAULT_DIMENSIONS
) -> Embedder:
    """Process-wide embedder singleton.

    Shared because the sentence-transformers backend is expensive to construct
    and holds a model in memory; the hashing one is stateless either way.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                _EMBEDDER = build_embedder(backend, dimensions)
    return _EMBEDDER


def reset_embedder() -> None:
    """Drop the cached embedder (used by tests that switch backends)."""
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        _EMBEDDER = None
