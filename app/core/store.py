"""Document lifecycle: ingest, tag, persist, reload, delete.

The corpus lives in a SQLite database (``corpus.db``) and the hybrid index is
rebuilt from it at boot. Embeddings are cached in the ``chunks`` table, keyed by
the embedder's name and dimensions, so switching backends invalidates them
rather than silently mixing incompatible vector spaces — vectors from two
different embedders are not comparable, and a stale cache would produce
confidently wrong rankings.

**Changed from the previous version: persistence is transactional.** The corpus
used to be a single JSON file rewritten in full on every mutation. That was
atomic per write and still lossy, because each worker process held its own copy
of the whole corpus: the second save overwrote the first with a snapshot that
never contained the first worker's document. Both uploads returned 200 and one
document was permanently gone.

Every mutation now touches only its own rows, inside a transaction, so
concurrent writers no longer clobber each other's work. See
``app/core/corpus_db.py`` for the reproduction and the reasoning.

**Still per-process: the search index.** Durability and index freshness are
different problems, and only the first is solved here. A document written by
one worker is safely on disk and will be loaded by every worker at its next
start, but another worker's live index does not learn about it until then.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.core.chunking import chunk_text
from app.core.corpus_db import CorpusDB, migrate_json_corpus
from app.core.embedding import Embedder, get_embedder
from app.core.extract import extract
from app.core.metadata import ScopeFilter, collect_facets, normalize_metadata
from app.core.retrieval import Chunk, HybridIndex, RRF_K_DEFAULT


logger = logging.getLogger(__name__)


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

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._corpus_path = self.data_dir / "corpus.json"
        self.db = CorpusDB(self.data_dir / "corpus.db")

        # An existing JSON corpus is imported once, on the first boot after the
        # upgrade. The file is kept as a backup and never read again.
        if self.db.document_count() == 0:
            migrate_json_corpus(self._corpus_path, self.db)

        self._load()

    @property
    def _embedding_signature(self) -> str:
        """Identifies the vector space, so cached vectors are never mismatched."""
        return "{}:{}".format(self.embedder.name, self.embedder.dimensions)

    # --- persistence ---------------------------------------------------

    def _load(self) -> None:
        """Rebuild in-memory state and the search index from the database.

        Each document is loaded independently: one bad row is skipped rather
        than discarding the rest of a salvageable corpus. That tolerance
        predates the database and is kept, because a corpus that mostly works
        is worth more than a clean failure at boot.
        """
        cached_signature = self.db.embedding_signature()
        reuse_vectors = cached_signature == self._embedding_signature

        rows = self.db.all_documents()
        if rows and not reuse_vectors:
            logger.info(
                "Embedding backend changed (%s -> %s); re-embedding the corpus.",
                cached_signature,
                self._embedding_signature,
            )

        for row in rows:
            try:
                self._load_one(row, reuse_vectors=reuse_vectors)
            except Exception:
                logger.warning("Skipping an unreadable document row.", exc_info=True)

        if not rows:
            # An empty corpus has no vectors to be stale, but the signature
            # must still be recorded: the first document written next will be
            # embedded by *this* embedder, and leaving the slot empty would
            # make the following boot think the space was unknown.
            if cached_signature is None:
                self.db.set_embedding_signature(self._embedding_signature)
            return

        if not reuse_vectors and self._documents:
            # The corpus has just been re-embedded in memory. Persist the new
            # vectors so the next boot is fast, writing only the embedding
            # column rather than rewriting the corpus.
            try:
                fresh = {
                    chunk.chunk_id: chunk.embedding
                    for chunk in self.index.all_chunks()
                }
                self.db.update_embeddings(fresh)
                self.db.set_embedding_signature(self._embedding_signature)
            except sqlite3.DatabaseError:
                logger.warning("Could not persist re-embedded vectors.", exc_info=True)
        elif cached_signature is None:
            self.db.set_embedding_signature(self._embedding_signature)

    def _load_one(self, row, reuse_vectors: bool = True) -> None:
        """Load one persisted document. Raises if the row is unusable."""
        doc = Document(
            doc_id=row["doc_id"],
            name=row["name"],
            size_bytes=int(row["size_bytes"]),
            chunk_count=int(row["chunk_count"]),
            char_count=int(row["char_count"]),
            sha256=row["sha256"],
            uploaded_at=row["uploaded_at"],
        )
        doc.metadata = normalize_metadata(json.loads(row["metadata"] or "{}"))

        # Validate every chunk before mutating any state, so a bad chunk midway
        # through cannot leave the document half-indexed.
        chunks = []
        for c in self.db.chunks_for(doc.doc_id):
            vector = None
            if reuse_vectors and c["embedding"]:
                candidate = json.loads(c["embedding"])
                if (
                    isinstance(candidate, list)
                    and len(candidate) == self.embedder.dimensions
                ):
                    vector = [float(v) for v in candidate]
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
            # Checked against the database, not the in-memory dict. Another
            # worker may have ingested these bytes since this process started,
            # and its document would be invisible to an in-memory scan. The
            # UNIQUE index on sha256 is the actual guarantee — this lookup
            # exists to raise the friendly error before hitting it.
            existing_row = self.db.find_by_sha256(digest)
            if existing_row is not None:
                raise DuplicateDocumentError(self._document_from_row(existing_row))

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

            # Indexed first so the chunks carry their embeddings, then written
            # in one transaction. If the write fails the index is rolled back
            # too, rather than leaving a document that is searchable in this
            # process and absent from every other one.
            self.index.add_chunks(chunks)
            try:
                self.db.insert_document(
                    doc.to_dict(),
                    [self._serialize_chunk(c) for c in chunks],
                )
            except sqlite3.IntegrityError as exc:
                self.index.remove_document(doc_id)
                # Lost the race to another worker between the check above and
                # this insert. The UNIQUE index caught it, which is the point
                # of enforcing it in the database rather than in Python.
                race_winner = self.db.find_by_sha256(digest)
                if race_winner is not None:
                    raise DuplicateDocumentError(
                        self._document_from_row(race_winner)
                    ) from exc
                raise
            except Exception:
                self.index.remove_document(doc_id)
                raise

            self._documents[doc_id] = doc
            return doc

    @staticmethod
    def _serialize_chunk(chunk: Chunk) -> Dict:
        return {
            "chunk_id": chunk.chunk_id,
            "position": chunk.position,
            "text": chunk.text,
            "embedding": chunk.embedding,
        }

    @staticmethod
    def _document_from_row(row) -> Document:
        doc = Document(
            doc_id=row["doc_id"],
            name=row["name"],
            size_bytes=int(row["size_bytes"]),
            chunk_count=int(row["chunk_count"]),
            char_count=int(row["char_count"]),
            sha256=row["sha256"],
            uploaded_at=row["uploaded_at"],
        )
        doc.metadata = normalize_metadata(json.loads(row["metadata"] or "{}"))
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
            # The UPDATE touches one row, so a concurrent upload or delete of a
            # different document cannot be clobbered by it.
            if not self.db.update_metadata(doc_id, clean):
                # Deleted by another worker between the lookup and here.
                self._documents.pop(doc_id, None)
                self.index.remove_document(doc_id)
                return None
            doc.metadata = clean
            self.index.update_metadata(doc_id, clean)
            return doc

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            # The database is the authority on existence: another worker may
            # have removed this document, and returning True for a no-op delete
            # would report success for work that did not happen.
            removed = self.db.delete_document(doc_id)
            self._documents.pop(doc_id, None)
            self.index.remove_document(doc_id)
            return removed

    def clear(self) -> None:
        with self._lock:
            self.db.clear()
            self._documents = {}
            self.index.clear()
