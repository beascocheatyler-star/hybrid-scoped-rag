"""Notebooks: the unit of isolation, backed by SQLite.

``NotebookStore`` implements the same interface as the JSON ``DocumentStore``
(``add``, ``list_documents``, ``facets``, ``count_matching``, ``index``, ...),
so ``ChatEngine`` and the scope gate work unchanged. Each store owns a private
``HybridIndex`` built only from its own notebook's rows. There is no shared
index to filter, so IDF statistics, coverage, facets and retrieval candidates
cannot leak between notebooks (invariants I1, I2).

``NotebookRegistry`` creates, lists, copies and deletes notebooks, and hands out
lazily built stores.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app.core.chunking import chunk_spans
from app.core.db import Database, pack_vector, unpack_vector, utcnow
from app.core.embedding import Embedder
from app.core.extract import ExtractionError, extract_doc, kind_for
from app.core.fetch import check_url, fetch_bytes, page_to_doc, youtube_doc, youtube_id
from app.core.metadata import ScopeFilter, collect_facets, normalize_metadata
from app.core.retrieval import RRF_K_DEFAULT, Chunk, HybridIndex
from app.core.store import Document, DuplicateDocumentError

logger = logging.getLogger(__name__)

DEFAULT_NOTEBOOK_ID = "default"
LOCAL_USER_ID = "local"
SOURCE_KINDS = ("file", "url", "youtube", "text", "audio", "image", "note")


class NotebookNotFound(KeyError):
    pass


class SourceLimitError(ValueError):
    pass


class JobCancelled(Exception):
    pass


@dataclass
class Notebook:
    id: str
    owner_id: str
    title: str
    description: str = ""
    emoji: str = ""
    chat_config: Dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row) -> "Notebook":
        try:
            cfg = json.loads(row["chat_config_json"] or "{}")
        except ValueError:
            cfg = {}
        return cls(
            id=row["id"], owner_id=row["owner_id"], title=row["title"],
            description=row["description"], emoji=row["emoji"], chat_config=cfg,
            created_at=row["created_at"], updated_at=row["updated_at"],
        )


@dataclass
class Source:
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
    has_original: bool = False
    guide: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_document(self) -> Document:
        return Document(
            doc_id=self.id, name=self.title, size_bytes=self.size_bytes,
            chunk_count=self.chunk_count, char_count=self.char_count,
            sha256=self.sha256, uploaded_at=self.created_at,
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_row(cls, row) -> "Source":
        try:
            meta = json.loads(row["metadata_json"] or "{}")
        except ValueError:
            meta = {}
        try:
            guide = json.loads(row["guide_json"] or "{}")
        except ValueError:
            guide = {}
        keys = row.keys()
        return cls(
            id=row["id"], notebook_id=row["notebook_id"], kind=row["kind"],
            title=row["title"], original_name=row["original_name"], mime=row["mime"],
            sha256=row["sha256"], size_bytes=row["size_bytes"], uri=row["uri"],
            status=row["status"], error=row["error"], metadata=meta,
            char_count=row["char_count"], chunk_count=row["chunk_count"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            has_original=bool(row["file_path"]) if "file_path" in keys else False,
            guide=guide,
        )


_SOURCE_SELECT = (
    "SELECT s.*, f.path AS file_path FROM sources s "
    "LEFT JOIN source_files f ON f.source_id = s.id "
)


class NotebookStore:
    """One notebook's sources, chunks and private hybrid index."""

    def __init__(
        self,
        db: Database,
        notebook_id: str,
        data_dir: Path,
        embedder: Embedder,
        chunk_size: int = 1200,
        overlap: int = 200,
        rrf_k: int = RRF_K_DEFAULT,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        max_sources: int = 50,
        max_source_words: int = 500_000,
    ):
        self.db = db
        self.notebook_id = notebook_id
        self.data_dir = Path(data_dir)
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_sources = max_sources
        self.max_source_words = max_source_words
        self.index = HybridIndex(
            embedder=embedder, rrf_k=rrf_k,
            lexical_weight=lexical_weight, dense_weight=dense_weight,
        )
        self._lock = threading.RLock()
        self._docs: Dict[str, Document] = {}
        self._load()

    # --- paths & identity ------------------------------------------------

    @property
    def _embedding_signature(self) -> str:
        return "{}:{}".format(self.embedder.name, self.embedder.dimensions)

    @property
    def notebook_dir(self) -> Path:
        return self.data_dir / "notebooks" / self.notebook_id

    def source_dir(self, source_id: str) -> Path:
        return self.notebook_dir / "sources" / source_id

    # --- loading ---------------------------------------------------------

    def _load(self) -> None:
        """Build the index from this notebook's ready sources only."""
        sig = self._embedding_signature
        sources = [
            Source.from_row(r)
            for r in self.db.all(
                _SOURCE_SELECT + "WHERE s.notebook_id = ? AND s.status = 'ready' "
                "ORDER BY s.created_at, s.id",
                (self.notebook_id,),
            )
        ]
        by_id = {s.id: s for s in sources}
        rows = self.db.all(
            "SELECT c.id, c.source_id, c.position, c.text, c.start_char, c.end_char, "
            "c.page, c.slide, c.t_start_ms, c.t_end_ms, c.section_path, v.vector, v.dim "
            "FROM chunks c JOIN sources s ON s.id = c.source_id "
            "LEFT JOIN chunk_vectors v ON v.chunk_id = c.id AND v.embedder_signature = ? "
            "WHERE c.notebook_id = ? AND s.status = 'ready' "
            "ORDER BY s.created_at, s.id, c.position",
            (sig, self.notebook_id),
        )
        chunks: List[Chunk] = []
        for r in rows:
            src = by_id.get(r["source_id"])
            if src is None:
                continue
            vec = None
            if r["vector"] is not None and r["dim"] == self.embedder.dimensions:
                vec = unpack_vector(r["vector"])
            chunks.append(Chunk(
                chunk_id=r["id"], doc_id=src.id, doc_name=src.title,
                position=r["position"], text=r["text"],
                metadata=dict(src.metadata), embedding=vec,
                start_char=r["start_char"], end_char=r["end_char"], page=r["page"],
                slide=r["slide"], t_start_ms=r["t_start_ms"], t_end_ms=r["t_end_ms"],
                section_path=r["section_path"],
            ))
        missing = [c for c in chunks if c.embedding is None]
        self.index.add_chunks(chunks)  # embeds any chunk without a vector
        if missing:
            logger.info(
                "Notebook %s: embedded %d chunks for signature %s",
                self.notebook_id, len(missing), sig,
            )
            with self.db.tx() as c:
                c.executemany(
                    "INSERT OR REPLACE INTO chunk_vectors"
                    "(chunk_id, embedder_signature, dim, vector) VALUES (?,?,?,?)",
                    [(ch.chunk_id, sig, len(ch.embedding), pack_vector(ch.embedding))
                     for ch in missing],
                )
        with self._lock:
            self._docs = {s.id: s.to_document() for s in sources}

    # --- queries (DocumentStore interface) -------------------------------

    def list_documents(self) -> List[Document]:
        with self._lock:
            return sorted(self._docs.values(), key=lambda d: (d.uploaded_at, d.doc_id))

    def get(self, doc_id: str) -> Optional[Document]:
        with self._lock:
            return self._docs.get(doc_id)

    def facets(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return collect_facets(d.metadata for d in self._docs.values())

    def count_matching(self, scope: Optional[ScopeFilter]) -> int:
        with self._lock:
            if scope is None or scope.is_empty:
                return len(self._docs)
            return sum(1 for d in self._docs.values() if scope.matches(d.doc_id, d.metadata))

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return not self._docs

    @property
    def total_chunks(self) -> int:
        return len(self.index)

    # --- sources (all statuses) ------------------------------------------

    def list_sources(self) -> List[Source]:
        return [
            Source.from_row(r)
            for r in self.db.all(
                _SOURCE_SELECT + "WHERE s.notebook_id = ? ORDER BY s.created_at, s.id",
                (self.notebook_id,),
            )
        ]

    def get_source(self, source_id: str) -> Optional[Source]:
        row = self.db.one(
            _SOURCE_SELECT + "WHERE s.notebook_id = ? AND s.id = ?",
            (self.notebook_id, source_id),
        )
        return Source.from_row(row) if row else None

    def original_path(self, source_id: str) -> Optional[Path]:
        row = self.db.one(
            "SELECT f.path FROM source_files f JOIN sources s ON s.id = f.source_id "
            "WHERE s.notebook_id = ? AND s.id = ?",
            (self.notebook_id, source_id),
        )
        if not row:
            return None
        p = self.data_dir / row["path"]
        return p if p.is_file() else None

    def extracted_text(self, source_id: str) -> Optional[str]:
        p = self.source_dir(source_id) / "extracted.txt"
        if p.is_file():
            return p.read_text("utf-8")
        # Imported legacy sources have no extracted text on disk; rebuild an
        # approximation from their chunks so the content endpoint still works.
        rows = self.db.all(
            "SELECT text FROM chunks WHERE source_id = ? AND notebook_id = ? ORDER BY position",
            (source_id, self.notebook_id),
        )
        return "\n\n".join(r["text"] for r in rows) if rows else None

    # --- mutation --------------------------------------------------------

    def create_source(
        self,
        filename: str,
        data: bytes,
        metadata: Optional[Dict[str, object]] = None,
        kind: str = "file",
        mime: str = "",
        title: Optional[str] = None,
        uri: str = "",
    ) -> Source:
        """Register a source as ``queued`` and store its original bytes.

        Raises DuplicateDocumentError, SourceLimitError, InvalidFilterError.
        Nothing is extracted yet: see ``process_source``.
        """
        if kind not in SOURCE_KINDS:
            raise ValueError("Unknown source kind: {}".format(kind))
        if kind == "file":
            kind = kind_for(filename)  # 'image' / 'audio' / 'file' 
        clean = normalize_metadata(metadata)
        digest = hashlib.sha256(data).hexdigest()
        safe_name = Path(filename).name or "unnamed"
        source_id = uuid.uuid4().hex[:12]
        rel = Path("notebooks") / self.notebook_id / "sources" / source_id / "original" / safe_name

        dup = self.db.one(
            _SOURCE_SELECT + "WHERE s.notebook_id = ? AND s.sha256 = ?",
            (self.notebook_id, digest),
        )
        if dup and dup["status"] == "failed":
            # A failed attempt must not block a retry of the same bytes.
            self._purge(dup["id"])
        with self.db.tx() as c:
            dup = c.execute(
                _SOURCE_SELECT + "WHERE s.notebook_id = ? AND s.sha256 = ?",
                (self.notebook_id, digest),
            ).fetchone()
            if dup:
                raise DuplicateDocumentError(Source.from_row(dup).to_document())
            self._check_limit(c)
            path = self.data_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            c.execute(
                "INSERT INTO sources(id, notebook_id, kind, title, original_name, mime, "
                "sha256, size_bytes, uri, status, metadata_json, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?, 'queued', ?, ?, ?)",
                (source_id, self.notebook_id, kind, title or safe_name, safe_name, mime,
                 digest, len(data), uri, json.dumps(clean), utcnow(), utcnow()),
            )
            c.execute(
                "INSERT INTO source_files(source_id, path) VALUES (?, ?)",
                (source_id, rel.as_posix()),
            )
        return self.get_source(source_id)  # type: ignore[return-value]

    def _set_status(self, source_id: str, status: str, error: str = "") -> None:
        self.db.execute(
            "UPDATE sources SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (status, error, utcnow(), source_id),
        )

    def process_source(
        self,
        source_id: str,
        cancelled: Optional[Callable[[], bool]] = None,
        progress: Optional[Callable[[float], None]] = None,
    ) -> Document:
        """Extract, chunk, embed, persist and index a queued source.

        On any failure the source is marked ``failed`` with the reason, and the
        exception propagates. ``cancelled`` is polled between stages.
        """
        def check():
            if cancelled and cancelled():
                raise JobCancelled()

        def step(p):
            if progress:
                progress(p)

        src = self.get_source(source_id)
        if src is None:
            raise KeyError(source_id)
        if src.status == "ready":
            return src.to_document()
        try:
            check()
            self._set_status(source_id, "processing")
            doc, title, uri = self._extract(src)
            words = len(doc.text.split())
            if self.max_source_words and words > self.max_source_words:
                raise ExtractionError(
                    "This source has {:,} words; the limit is {:,} (MAX_SOURCE_WORDS)."
                    .format(words, self.max_source_words))
            step(0.3)
            check()
            spans = chunk_spans(doc.text, self.chunk_size, self.overlap)
            chunks = []
            for i, sp in enumerate(spans):
                t0, t1 = doc.time_span(sp.start, sp.end)
                chunks.append(Chunk(
                    chunk_id="{}-{}".format(source_id, i), doc_id=source_id,
                    doc_name=title, position=i, text=sp.text,
                    metadata=dict(src.metadata), start_char=sp.start, end_char=sp.end,
                    page=doc.page_at(sp.start), slide=doc.slide_at(sp.start),
                    t_start_ms=t0, t_end_ms=t1, section_path=doc.section_at(sp.start),
                ))
            vectors = self.embedder.embed_batch([c.text for c in chunks]) if chunks else []
            for ch, v in zip(chunks, vectors):
                ch.embedding = v
            step(0.8)
            check()
            sdir = self.source_dir(source_id)
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / "extracted.txt").write_text(doc.text, "utf-8")
            sig = self._embedding_signature
            now = utcnow()
            with self.db.tx() as c:
                c.executemany(
                    "INSERT INTO chunks(id, source_id, notebook_id, position, text, "
                    "start_char, end_char, page, slide, t_start_ms, t_end_ms, section_path, "
                    "token_count, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(ch.chunk_id, source_id, self.notebook_id, ch.position, ch.text,
                      ch.start_char, ch.end_char, ch.page, ch.slide, ch.t_start_ms,
                      ch.t_end_ms, ch.section_path, ch.length, now, now) for ch in chunks],
                )
                c.executemany(
                    "INSERT INTO chunk_vectors(chunk_id, embedder_signature, dim, vector) "
                    "VALUES (?,?,?,?)",
                    [(ch.chunk_id, sig, len(ch.embedding), pack_vector(ch.embedding))
                     for ch in chunks],
                )
                c.execute(
                    "UPDATE sources SET status = 'ready', error = '', char_count = ?, "
                    "chunk_count = ?, title = ?, uri = ?, updated_at = ? WHERE id = ?",
                    (len(doc.text), len(chunks), title, uri, now, source_id),
                )
            self.index.add_chunks(chunks)
            ready = self.get_source(source_id).to_document()  # type: ignore[union-attr]
            with self._lock:
                self._docs[source_id] = ready
            step(1.0)
            return ready
        except JobCancelled:
            self._set_status(source_id, "failed", "Cancelled.")
            raise
        except Exception as exc:
            self._set_status(source_id, "failed", str(exc) or exc.__class__.__name__)
            raise

    def _extract(self, src: "Source"):
        """Return (Extracted, title, uri) for any source kind."""
        if src.kind in ("url", "youtube"):
            if src.kind == "youtube":
                doc = youtube_doc(src.uri)
                uri = src.uri
                raw, name = doc.text.encode("utf-8"), "transcript.txt"
            else:
                fetched = fetch_bytes(src.uri)
                doc, uri = page_to_doc(fetched)
                raw, name = fetched.data, "page.html" if "html" in fetched.content_type else "page.bin"
            # Keep what was fetched as the source's "original".
            rel = Path("notebooks") / self.notebook_id / "sources" / src.id / "original" / name
            (self.data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.data_dir / rel).write_bytes(raw)
            self.db.execute(
                "INSERT OR REPLACE INTO source_files(source_id, path) VALUES (?, ?)",
                (src.id, rel.as_posix()))
            title = src.title
            if doc.title and (not src.title or src.title == src.uri):
                title = doc.title[:200]
            return doc, title, uri
        path = self.original_path(src.id)
        if path is None:
            raise FileNotFoundError("The original file for this source is missing.")
        return extract_doc(src.original_name, path.read_bytes()), src.title, src.uri

    def create_remote_source(self, kind: str, url: str, title: str = "",
                             metadata: Optional[Dict[str, object]] = None) -> "Source":
        """Register a url or youtube source. The fetch happens in the job; the
        URL is validated (SSRF guard) up front so bad input fails fast."""
        url = url.strip()
        if kind == "url":
            check_url(url)
        elif kind == "youtube":
            youtube_id(url)
        else:
            raise ValueError("Unknown remote kind {!r}".format(kind))
        clean = normalize_metadata(metadata)
        digest = hashlib.sha256("{}:{}".format(kind, url).encode()).hexdigest()
        dup = self.db.one(_SOURCE_SELECT + "WHERE s.notebook_id = ? AND s.sha256 = ?",
                          (self.notebook_id, digest))
        if dup and dup["status"] == "failed":
            self._purge(dup["id"])
        source_id = uuid.uuid4().hex[:12]
        with self.db.tx() as c:
            dup = c.execute(_SOURCE_SELECT + "WHERE s.notebook_id = ? AND s.sha256 = ?",
                            (self.notebook_id, digest)).fetchone()
            if dup:
                raise DuplicateDocumentError(Source.from_row(dup).to_document())
            self._check_limit(c)
            c.execute(
                "INSERT INTO sources(id, notebook_id, kind, title, original_name, sha256, "
                "uri, status, metadata_json, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?, 'queued', ?, ?, ?)",
                (source_id, self.notebook_id, kind, (title or url)[:200], "", digest, url,
                 json.dumps(clean), utcnow(), utcnow()))
        return self.get_source(source_id)  # type: ignore[return-value]

    def _check_limit(self, c) -> None:
        n = c.execute("SELECT COUNT(*) FROM sources WHERE notebook_id = ?",
                      (self.notebook_id,)).fetchone()[0]
        if self.max_sources and n >= self.max_sources:
            raise SourceLimitError(
                "This notebook already has {} sources; the limit is {} "
                "(MAX_SOURCES_PER_NOTEBOOK).".format(n, self.max_sources))

    def source_chunks(self, source_id: str) -> List[Dict]:
        return [dict(r) for r in self.db.all(
            "SELECT id, position, text, start_char, end_char, page, slide, t_start_ms, "
            "t_end_ms, section_path FROM chunks WHERE source_id = ? AND notebook_id = ? "
            "ORDER BY position", (source_id, self.notebook_id))]

    def set_guide(self, source_id: str, guide: Dict) -> None:
        self.db.execute(
            "UPDATE sources SET guide_json = ?, updated_at = ? WHERE id = ? AND notebook_id = ?",
            (json.dumps(guide, ensure_ascii=False), utcnow(), source_id, self.notebook_id))

    def apply_auto_labels(self, labels: Dict[str, str]) -> int:
        """Write metadata["label"] unless the user changed it since the last
        automatic labelling. Returns the number of sources updated."""
        changed = 0
        for src in self.list_sources():
            new = labels.get(src.id)
            if not new or src.status != "ready":
                continue
            last_auto = src.guide.get("auto_label")
            current = src.metadata.get("label")
            if current is not None and current != last_auto:
                continue  # user-edited label: leave it alone
            meta = dict(src.metadata)
            meta["label"] = new
            doc = self.set_metadata(src.id, meta)
            guide = dict(src.guide)
            guide["auto_label"] = doc.metadata.get("label") if doc else new
            self.set_guide(src.id, guide)
            changed += 1
        return changed

    def chunk_row(self, chunk_id: str):
        return self.db.one(
            "SELECT * FROM chunks WHERE id = ? AND notebook_id = ?", (chunk_id, self.notebook_id))

    def add(
        self, filename: str, data: bytes, metadata: Optional[Dict[str, object]] = None
    ) -> Document:
        """Synchronous ingest (the v1 contract). A file that cannot be
        extracted leaves nothing behind, matching the JSON store."""
        src = self.create_source(filename, data, metadata)
        try:
            return self.process_source(src.id)
        except Exception:
            self._purge(src.id)
            raise

    def set_metadata(
        self, doc_id: str, metadata: Optional[Dict[str, object]]
    ) -> Optional[Document]:
        clean = normalize_metadata(metadata)
        cur = self.db.execute(
            "UPDATE sources SET metadata_json = ?, updated_at = ? "
            "WHERE id = ? AND notebook_id = ?",
            (json.dumps(clean), utcnow(), doc_id, self.notebook_id),
        )
        if cur.rowcount == 0:
            return None
        self.index.update_metadata(doc_id, clean)
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is not None:
                doc.metadata = clean
                return doc
        src = self.get_source(doc_id)
        return src.to_document() if src else None

    def rename(self, source_id: str, title: str) -> Optional[Source]:
        title = " ".join((title or "").split())[:200]
        if not title:
            raise ValueError("Title must not be empty.")
        cur = self.db.execute(
            "UPDATE sources SET title = ?, updated_at = ? WHERE id = ? AND notebook_id = ?",
            (title, utcnow(), source_id, self.notebook_id),
        )
        if cur.rowcount == 0:
            return None
        with self._lock:
            if source_id in self._docs:
                self._docs[source_id].name = title
                for ch in self.index._chunks:
                    if ch.doc_id == source_id:
                        ch.doc_name = title
        return self.get_source(source_id)

    def _purge(self, source_id: str) -> None:
        self.db.execute(
            "DELETE FROM sources WHERE id = ? AND notebook_id = ?",
            (source_id, self.notebook_id),
        )
        self.index.remove_document(source_id)
        with self._lock:
            self._docs.pop(source_id, None)
        shutil.rmtree(self.source_dir(source_id), ignore_errors=True)

    def delete(self, doc_id: str) -> bool:
        if self.get_source(doc_id) is None:
            return False
        self._purge(doc_id)
        return True

    def clear(self) -> None:
        for s in self.list_sources():
            self._purge(s.id)


class NotebookRegistry:
    """Creates and finds notebooks; caches one ``NotebookStore`` per notebook."""

    def __init__(self, db: Database, data_dir: Path, embedder: Embedder, **store_kwargs):
        self.db = db
        self.data_dir = Path(data_dir)
        self.embedder = embedder
        self.store_kwargs = store_kwargs
        self._stores: Dict[str, NotebookStore] = {}
        self._lock = threading.RLock()
        self._ensure_bootstrap()

    def _ensure_bootstrap(self) -> None:
        with self.db.tx() as c:
            c.execute(
                "INSERT OR IGNORE INTO users(id, display_name) VALUES (?, 'Local user')",
                (LOCAL_USER_ID,),
            )
            if not c.execute(
                "SELECT 1 FROM notebooks WHERE id = ?", (DEFAULT_NOTEBOOK_ID,)
            ).fetchone():
                self._insert(c, DEFAULT_NOTEBOOK_ID, LOCAL_USER_ID, "Default notebook",
                             "Holds everything uploaded through the v1 API.", "")

    @staticmethod
    def _insert(c, nb_id, owner_id, title, description, emoji, chat_config=None):
        now = utcnow()
        c.execute(
            "INSERT INTO notebooks(id, owner_id, title, description, emoji, "
            "chat_config_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (nb_id, owner_id, title, description, emoji,
             json.dumps(chat_config or {}), now, now),
        )
        c.execute(
            "INSERT INTO memberships(notebook_id, user_id, role) VALUES (?, ?, 'owner')",
            (nb_id, owner_id),
        )

    # --- notebooks -------------------------------------------------------

    def create(self, title: str, description: str = "", emoji: str = "",
               owner_id: str = LOCAL_USER_ID) -> Notebook:
        title = " ".join((title or "").split())[:200] or "Untitled notebook"
        nb_id = uuid.uuid4().hex[:12]
        with self.db.tx() as c:
            self._insert(c, nb_id, owner_id, title, description[:2000], emoji[:16])
        return self.get(nb_id)

    def get(self, nb_id: str) -> Notebook:
        row = self.db.one("SELECT * FROM notebooks WHERE id = ?", (nb_id,))
        if not row:
            raise NotebookNotFound(nb_id)
        return Notebook.from_row(row)

    def list(self, user_id: str = LOCAL_USER_ID) -> List[Notebook]:
        rows = self.db.all(
            "SELECT n.* FROM notebooks n JOIN memberships m ON m.notebook_id = n.id "
            "WHERE m.user_id = ? ORDER BY n.created_at, n.id",
            (user_id,),
        )
        return [Notebook.from_row(r) for r in rows]

    def update(self, nb_id: str, title: Optional[str] = None,
               description: Optional[str] = None, emoji: Optional[str] = None,
               chat_config: Optional[Dict] = None) -> Notebook:
        self.get(nb_id)
        sets, params = [], []
        if title is not None:
            t = " ".join(title.split())[:200]
            if not t:
                raise ValueError("Title must not be empty.")
            sets.append("title = ?"); params.append(t)
        if description is not None:
            sets.append("description = ?"); params.append(description[:2000])
        if emoji is not None:
            sets.append("emoji = ?"); params.append(emoji[:16])
        if chat_config is not None:
            sets.append("chat_config_json = ?"); params.append(json.dumps(chat_config))
        if sets:
            sets.append("updated_at = ?"); params.append(utcnow())
            self.db.execute(
                "UPDATE notebooks SET {} WHERE id = ?".format(", ".join(sets)),
                params + [nb_id],
            )
        return self.get(nb_id)

    def delete(self, nb_id: str) -> None:
        if nb_id == DEFAULT_NOTEBOOK_ID:
            raise ValueError(
                "The Default notebook backs the v1 API and cannot be deleted. "
                "Clear its sources instead."
            )
        self.get(nb_id)
        self.db.execute("DELETE FROM notebooks WHERE id = ?", (nb_id,))
        with self._lock:
            self._stores.pop(nb_id, None)
        shutil.rmtree(self.data_dir / "notebooks" / nb_id, ignore_errors=True)

    def copy(self, nb_id: str, title: Optional[str] = None,
             owner_id: str = LOCAL_USER_ID) -> Notebook:
        """Copy a notebook's ready sources (rows, chunks, vectors, originals).

        Chats and notes are not copied, matching NotebookLM. Artifacts arrive in
        M6 and will be copied then.
        """
        src_nb = self.get(nb_id)
        new = self.create(title or "Copy of {}".format(src_nb.title),
                          src_nb.description, src_nb.emoji, owner_id)
        self.db.execute(
            "UPDATE notebooks SET chat_config_json = ? WHERE id = ?",
            (json.dumps(src_nb.chat_config), new.id),
        )
        sources = self.db.all(
            "SELECT * FROM sources WHERE notebook_id = ? AND status = 'ready'", (nb_id,)
        )
        for s in sources:
            new_sid = uuid.uuid4().hex[:12]
            old_dir = self.data_dir / "notebooks" / nb_id / "sources" / s["id"]
            new_dir = self.data_dir / "notebooks" / new.id / "sources" / new_sid
            with self.db.tx() as c:
                cols = [k for k in s.keys() if k not in ("id", "notebook_id")]
                c.execute(
                    "INSERT INTO sources(id, notebook_id, {}) SELECT ?, ?, {} "
                    "FROM sources WHERE id = ?".format(", ".join(cols), ", ".join(cols)),
                    (new_sid, new.id, s["id"]),
                )
                f = c.execute("SELECT path FROM source_files WHERE source_id = ?",
                              (s["id"],)).fetchone()
                if f:
                    rel = Path(f["path"])
                    new_rel = Path("notebooks") / new.id / "sources" / new_sid / rel.relative_to(
                        Path("notebooks") / nb_id / "sources" / s["id"])
                    c.execute("INSERT INTO source_files(source_id, path) VALUES (?, ?)",
                              (new_sid, new_rel.as_posix()))
                chunk_rows = c.execute(
                    "SELECT * FROM chunks WHERE source_id = ? ORDER BY position", (s["id"],)
                ).fetchall()
                for ch in chunk_rows:
                    new_cid = "{}-{}".format(new_sid, ch["position"])
                    ccols = [k for k in ch.keys() if k not in ("id", "source_id", "notebook_id")]
                    c.execute(
                        "INSERT INTO chunks(id, source_id, notebook_id, {}) SELECT ?, ?, ?, {} "
                        "FROM chunks WHERE id = ?".format(", ".join(ccols), ", ".join(ccols)),
                        (new_cid, new_sid, new.id, ch["id"]),
                    )
                    c.execute(
                        "INSERT INTO chunk_vectors(chunk_id, embedder_signature, dim, vector) "
                        "SELECT ?, embedder_signature, dim, vector FROM chunk_vectors "
                        "WHERE chunk_id = ?",
                        (new_cid, ch["id"]),
                    )
            if old_dir.exists():
                shutil.copytree(old_dir, new_dir)
        return self.get(new.id)

    # --- stores ----------------------------------------------------------

    def store(self, nb_id: str) -> NotebookStore:
        with self._lock:
            st = self._stores.get(nb_id)
            if st is None:
                self.get(nb_id)  # raises NotebookNotFound
                kwargs = dict(self.store_kwargs)
                if nb_id == DEFAULT_NOTEBOOK_ID:
                    # The v1 API never had a source limit; imposing one on the
                    # notebook that backs it would break existing deployments.
                    kwargs["max_sources"] = 0
                st = NotebookStore(self.db, nb_id, self.data_dir, self.embedder, **kwargs)
                self._stores[nb_id] = st
            return st

    def default_store(self) -> NotebookStore:
        return self.store(DEFAULT_NOTEBOOK_ID)
