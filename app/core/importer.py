"""One-time import of a pre-M1 ``corpus.json`` / ``persona.json``.

The legacy file is read through the existing JSON ``DocumentStore``, so every
tolerance that store already has (and is already tested for) applies here too:
version 1 and 2 corpora, corrupt JSON, malformed entries skipped individually,
re-embedding when the embedder changed.

Order of operations, chosen so that no failure can lose data:

1. Copy ``corpus.json`` and ``persona.json`` into
   ``legacy-backup-<timestamp>/``.
2. Load the legacy corpus and write it into the Default notebook in one
   transaction per document.
3. Remove the live ``corpus.json`` (its pristine copy is in the backup
   folder), so a second boot cannot import it twice, and record completion
   in ``app_meta``.

``persona.json`` stays in place because the v1 persona routes still read it.
A copy goes into the Default notebook's ``chat_config_json`` for M4.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from app.core.db import Database, pack_vector, utcnow
from app.core.embedding import Embedder
from app.core.notebooks import DEFAULT_NOTEBOOK_ID, NotebookRegistry
from app.core.store import DocumentStore

logger = logging.getLogger(__name__)

META_KEY = "legacy_import"


def import_legacy(
    data_dir: Path, db: Database, registry: NotebookRegistry, embedder: Embedder
) -> Optional[Dict]:
    """Import once. Returns a summary, or None if there was nothing to do."""
    data_dir = Path(data_dir)
    corpus = data_dir / "corpus.json"
    persona = data_dir / "persona.json"
    if db.get_meta(META_KEY) or not corpus.exists():
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = data_dir / "legacy-backup-{}".format(stamp)
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corpus, backup / "corpus.json")
    if persona.exists():
        shutil.copy2(persona, backup / "persona.json")

    # Load from a scratch copy: DocumentStore may rewrite the file it reads
    # (after re-embedding), and the backup must stay byte-identical.
    with tempfile.TemporaryDirectory(prefix="hsr-import-") as tmp:
        shutil.copy2(corpus, Path(tmp) / "corpus.json")
        legacy = DocumentStore(data_dir=Path(tmp), embedder=embedder)
        sig = "{}:{}".format(embedder.name, embedder.dimensions)
        chunks_by_doc: Dict[str, list] = {}
        for ch in legacy.index._chunks:
            chunks_by_doc.setdefault(ch.doc_id, []).append(ch)

        imported, skipped = 0, 0
        for doc in legacy.list_documents():
            try:
                with db.tx() as c:
                    if c.execute(
                        "SELECT 1 FROM sources WHERE notebook_id = ? AND sha256 = ?",
                        (DEFAULT_NOTEBOOK_ID, doc.sha256),
                    ).fetchone():
                        skipped += 1
                        continue
                    chunks = sorted(chunks_by_doc.get(doc.doc_id, []), key=lambda x: x.position)
                    c.execute(
                        "INSERT INTO sources(id, notebook_id, kind, title, original_name, "
                        "sha256, size_bytes, status, metadata_json, char_count, chunk_count, "
                        "created_at, updated_at) VALUES (?,?, 'file', ?, ?, ?, ?, 'ready', ?, ?, ?, ?, ?)",
                        (doc.doc_id, DEFAULT_NOTEBOOK_ID, doc.name, doc.name, doc.sha256,
                         doc.size_bytes, json.dumps(doc.metadata), doc.char_count,
                         len(chunks), doc.uploaded_at or utcnow(), utcnow()),
                    )
                    for ch in chunks:
                        c.execute(
                            "INSERT INTO chunks(id, source_id, notebook_id, position, text, "
                            "token_count) VALUES (?,?,?,?,?,?)",
                            (ch.chunk_id, doc.doc_id, DEFAULT_NOTEBOOK_ID, ch.position,
                             ch.text, ch.length),
                        )
                        if ch.embedding is not None:
                            c.execute(
                                "INSERT INTO chunk_vectors(chunk_id, embedder_signature, dim, vector) "
                                "VALUES (?,?,?,?)",
                                (ch.chunk_id, sig, len(ch.embedding), pack_vector(ch.embedding)),
                            )
                imported += 1
            except Exception:
                logger.warning("Could not import legacy document %s", doc.name, exc_info=True)
                skipped += 1

    if persona.exists():
        try:
            cfg = json.loads(persona.read_text("utf-8"))
            if isinstance(cfg, dict):
                registry.update(DEFAULT_NOTEBOOK_ID, chat_config={"persona": cfg})
        except (ValueError, OSError):
            logger.warning("persona.json unreadable; not copied into the notebook.")

    # The pristine copy is already in the backup folder; removing the live file
    # is what makes the import happen exactly once.
    corpus.unlink()
    summary = {"imported": imported, "skipped": skipped, "backup": backup.name, "at": utcnow()}
    db.set_meta(META_KEY, json.dumps(summary))
    # Drop any cached store so the Default notebook reloads with the import.
    with registry._lock:
        registry._stores.pop(DEFAULT_NOTEBOOK_ID, None)
    logger.info("Imported %d legacy document(s) into the Default notebook (backup: %s)",
                imported, backup.name)
    return summary
