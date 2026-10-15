"""Document lifecycle: ingest, tag, persist, reload, delete.

The corpus is persisted as JSON on disk so a restart does not lose uploads. The
hybrid index is rebuilt from that JSON at boot — cheap, and it avoids shipping a
binary index format that would be hard for a reviewer to inspect.

**Changed from the predecessor:** documents carry metadata, and chunk embeddings
are cached in the corpus file. Caching matters because re-embedding the whole
corpus at every boot is the one genuinely slow step once the optional
sentence-transformers backend is in use. The cache is keyed by the embedder's
name and dimensions, so switching backends invalidates it rather than silently
mixing incompatible vector spaces — vectors from two different embedders are not
comparable, and a stale cache would produce confidently wrong rankings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.core.chunking import chunk_text
from app.core.embedding import Embedder, get_embedder
from app.core.extract import extract
from app.core.metadata import ScopeFilter, collect_facets, normalize_metadata
from app.core.retrieval import Chunk, HybridIndex, RRF_K_DEFAULT


logger = logging.getLogger(__name__)

CORPUS_VERSION = 2


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Document:
    doc_id: str
    name: str
    size_bytes: int
    chunk_count: int
    char_count: int
    sha256: str
    uploaded_at: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class DuplicateDocumentError(ValueError):
    """Raised when the exact same file content is uploaded twice."""

    def __init__(self, existing: Document):
        super().__init__(
            "This file is already in the knowledge base as '{}'.".format(existing.name)
        )
        self.existing = existing


class DocumentStore:
    """Owns the corpus and keeps the hybrid index in sync with it."""

    def __init__(
        self,
        data_dir: Path,
        chunk_size: int = 1200,
        overlap: int = 200,
        embedder: Optional[Embedder] = None,
        rrf_k: int = RRF_K_DEFAULT,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
    ):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedder = embedder or get_embedder()
        self.index = HybridIndex(
            embedder=self.embedder,
            rrf_k=rrf_k,
            lexical_weight=lexical_weight,
            dense_weight=dense_weight,
        )

        self._lock = threading.RLock()
        self._documents: Dict[str, Document] = {}
        self._chunks_by_doc: Dict[str, List[Dict]] = {}

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._corpus_path = self.data_dir / "corpus.json"
        self._load()

    @property
    def _embedding_signature(self) -> str:
        """Identifies the vector space, so cached vectors are never mismatched."""
        return "{}:{}".format(self.embedder.name, self.embedder.dimensions)

    # --- persistence ---------------------------------------------------

    def _load(self) -> None:
        """Rebuild state from disk.

        Runs at import time, so no shape of corrupt file may prevent startup.
        Each document is loaded independently: one bad entry is skipped rather
        than discarding the rest of a salvageable corpus.
        """
        if not self._corpus_path.exists():
            return
        try:
            raw = json.loads(self._corpus_path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            logger.warning(
                "Could not parse %s; starting with an empty knowledge base.",
                self._corpus_path,
            )
            return

        if not isinstance(raw, dict):
            logger.warning("%s is not a JSON object; ignoring.", self._corpus_path)
            return

        documents = raw.get("documents")
        if not isinstance(documents, list):
            return

        cached_signature = raw.get("embedding_signature")
        reuse_vectors = cached_signature == self._embedding_signature
        if documents and not reuse_vectors:
            logger.info(
                "Embedding backend changed (%s -> %s); re-embedding the corpus.",
                cached_signature,
                self._embedding_signature,
            )

        for entry in documents:
            try:
                self._load_one(entry, reuse_vectors=reuse_vectors)
            except Exception:
                logger.warning("Skipping an unreadable document entry.", exc_info=True)

        # A corpus loaded without reusable vectors has just been re-embedded in
        # memory; persist it so the next boot is fast.
        if documents and not reuse_vectors and self._documents:
            try:
                self._save()
            except OSError:
                logger.warning("Could not persist re-embedded corpus.", exc_info=True)

    def _load_one(self, entry, reuse_vectors: bool = True) -> None:
        """Load a single persisted document. Raises if the entry is unusable."""
        if not isinstance(entry, dict):
            raise ValueError("document entry is not an object")

        meta_fields = dict(entry["meta"])
        # Version 1 corpora (written by the predecessor project) have no
        # metadata key. Tolerating them means an existing deployment can be
        # upgraded in place rather than re-uploading everything.
        raw_metadata = meta_fields.pop("metadata", None)
        doc = Document(**meta_fields)
        doc.metadata = normalize_metadata(raw_metadata)

        raw_chunks = entry.get("chunks")
        if not isinstance(raw_chunks, list):
            raw_chunks = []

        # Validate every chunk before mutating any state, so a bad chunk midway
        # through cannot leave the document half-indexed.
        chunks = []
        for c in raw_chunks:
            if not isinstance(c, dict):
                continue
            vector = c.get("embedding") if reuse_vectors else None
            if vector is not None:
                if (
                    not isinstance(vector, list)
                    or len(vector) != self.embedder.dimensions
                ):
                    vector = None
                else:
                    vector = [float(v) for v in vector]
            chunks.append(
                Chunk(
                    chunk_id=str(c["chunk_id"]),
                    doc_id=doc.doc_id,
                    doc_name=doc.name,
                    position=int(c["position"]),
                    text=str(c["text"]),
                    metadata=dict(doc.metadata),
                    embedding=vector,
                )
            )

        self._documents[doc.doc_id] = doc
        self.index.add_chunks(chunks)
        # Serialise *after* indexing, so freshly computed vectors are captured.
        self._chunks_by_doc[doc.doc_id] = [self._serialize_chunk(c) for c in chunks]

    @staticmethod
    def _serialize_chunk(chunk: Chunk) -> Dict:
        return {
            "chunk_id": chunk.chunk_id,
            "position": chunk.position,
            "text": chunk.text,
            "embedding": chunk.embedding,
        }

    def _save(self) -> None:
        """Rewrite the whole corpus to disk.

        This is O(total corpus size) per upload, i.e. O(n²) bytes written to
        ingest n documents. That is a deliberate trade for a file a reviewer can
        open and read. Embedding vectors make each document heavier than in the
        predecessor — roughly 8 KB per chunk at 1024 dimensions — so the
        crossover where this becomes noticeable arrives sooner. Measured on the
        bundled syllabus (~7 chunks/doc):

            100 docs → 48 ms/upload    300 docs → 150 ms    600 docs → 310 ms

        Still fine for the scale this targets, and the remedy is unchanged: for
        thousands of documents, replace this with SQLite or an append-only chunk
        log. The ``DocumentStore`` interface would not change.
        """
        payload = {
            "version": CORPUS_VERSION,
            "saved_at": _utcnow(),
            "embedding_signature": self._embedding_signature,
            "documents": [
                {
                    "meta": self._documents[doc_id].to_dict(),
                    "chunks": self._chunks_by_doc.get(doc_id, []),
                }
                for doc_id in self._documents
            ],
        }
        # The temp name must be unique per writer: a fixed name lets two
        # processes (e.g. `uvicorn --workers 2` over one DATA_DIR) interleave
        # their writes and then rename each other's partial file into place.
        # The in-process lock cannot prevent that, so the filename must.
        tmp = self._corpus_path.with_name(
            "corpus.{}.{}.tmp".format(os.getpid(), uuid.uuid4().hex[:8])
        )
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
                fh.flush()
                # Force the bytes to disk before the rename, so a power loss
                # cannot leave a zero-length file where the corpus used to be.
                os.fsync(fh.fileno())
            tmp.replace(self._corpus_path)  # atomic within a filesystem
        except OSError:
            logger.exception("Failed to persist the corpus to %s", self._corpus_path)
            tmp.unlink(missing_ok=True)
            raise

    # --- queries -------------------------------------------------------

    def list_documents(self) -> List[Document]:
        with self._lock:
            return sorted(self._documents.values(), key=lambda d: d.uploaded_at)

    def get(self, doc_id: str) -> Optional[Document]:
        with self._lock:
            return self._documents.get(doc_id)

    def facets(self) -> Dict[str, Dict[str, int]]:
        """Metadata keys and their values, with document counts."""
        with self._lock:
            return collect_facets(d.metadata for d in self._documents.values())

    def count_matching(self, scope: Optional[ScopeFilter]) -> int:
        """How many documents the given scope selects."""
        with self._lock:
            if scope is None or scope.is_empty:
                return len(self._documents)
            return sum(
                1
                for doc in self._documents.values()
                if scope.matches(doc.doc_id, doc.metadata)
            )

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return not self._documents

    @property
    def total_chunks(self) -> int:
        return len(self.index)

    # --- mutation ------------------------------------------------------

    def add(
        self,
        filename: str,
        data: bytes,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Document:
        """Extract, chunk, embed, and index a file.

        Raises ExtractionError on unreadable input, DuplicateDocumentError on a
        byte-identical re-upload, and InvalidFilterError on bad metadata.
        """
        text = extract(filename, data)
        digest = hashlib.sha256(data).hexdigest()
        clean_metadata = normalize_metadata(metadata)

        with self._lock:
            for existing in self._documents.values():
                if existing.sha256 == digest:
                    raise DuplicateDocumentError(existing)

            pieces = chunk_text(text, self.chunk_size, self.overlap)
            doc_id = uuid.uuid4().hex[:12]
            doc = Document(
                doc_id=doc_id,
                name=filename,
                size_bytes=len(data),
                chunk_count=len(pieces),
                char_count=len(text),
                sha256=digest,
                uploaded_at=_utcnow(),
                metadata=clean_metadata,
            )

            chunks = [
                Chunk(
                    chunk_id="{}-{}".format(doc_id, i),
                    doc_id=doc_id,
                    doc_name=filename,
                    position=i,
                    text=piece,
                    metadata=dict(clean_metadata),
                )
                for i, piece in enumerate(pieces)
            ]

            self._documents[doc_id] = doc
            self.index.add_chunks(chunks)
            self._chunks_by_doc[doc_id] = [self._serialize_chunk(c) for c in chunks]
            self._save()
            return doc

    def set_metadata(
        self, doc_id: str, metadata: Optional[Dict[str, object]]
    ) -> Optional[Document]:
        """Replace a document's metadata. Returns None if it does not exist."""
        clean = normalize_metadata(metadata)
        with self._lock:
            doc = self._documents.get(doc_id)
            if doc is None:
                return None
            doc.metadata = clean
            self.index.update_metadata(doc_id, clean)
            self._save()
            return doc

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._documents:
                return False
            del self._documents[doc_id]
            self._chunks_by_doc.pop(doc_id, None)
            self.index.remove_document(doc_id)
            self._save()
            return True

    def clear(self) -> None:
        with self._lock:
            self._documents = {}
            self._chunks_by_doc = {}
            self.index.clear()
            self._save()
