"""Wire the storage layer together: database, notebooks, legacy import, jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from app.config import Settings
from app.core.db import Database
from app.core.embedding import Embedder
from app.core.importer import import_legacy
from app.core.jobs import Job, JobContext, JobQueue
from app.core.notebooks import NotebookRegistry


@dataclass
class Services:
    settings: Settings
    db: Database
    registry: NotebookRegistry
    jobs: JobQueue
    embedder: Embedder
    import_summary: Optional[Dict] = None

    def close(self) -> None:
        self.jobs.shutdown(wait=True)
        self.db.close()


def build_services(settings: Settings, embedder: Embedder, recover_jobs: bool = True) -> Services:
    settings.ensure_dirs()
    db = Database(settings.db_path)
    registry = NotebookRegistry(
        db,
        settings.data_dir,
        embedder,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        rrf_k=settings.rrf_k,
        lexical_weight=settings.lexical_weight,
        dense_weight=settings.dense_weight,
        max_sources=settings.max_sources_per_notebook,
    )
    summary = import_legacy(settings.data_dir, db, registry, embedder)

    jobs = JobQueue(db, workers=settings.job_workers)

    def ingest(job: Job, ctx: JobContext) -> Dict:
        store = registry.store(job.notebook_id)  # type: ignore[arg-type]
        doc = store.process_source(
            job.payload["source_id"], cancelled=ctx.cancelled, progress=ctx.progress
        )
        return {"source_id": doc.doc_id, "chunk_count": doc.chunk_count}

    jobs.register("ingest", ingest)
    if recover_jobs:
        jobs.recover()
    return Services(settings, db, registry, jobs, embedder, summary)
