"""Wire the storage layer together: database, notebooks, legacy import, jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from app.config import Settings
from app.core.db import Database
from app.core.embedding import Embedder
from app.core.guide import AUTO_LABEL_MIN, build_guide, compute_labels
from app.core.importer import import_legacy
from app.core.llm import get_llm
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


def apply_calibration(settings: Settings, db: Database, embedder: Embedder) -> Optional[Dict]:
    """Gate floors: explicit env var > measured calibration row > default.

    A non-hashing embedder without a calibration row gets dense rescue
    disabled: its similarity scale is unknown, and a floor tuned for hashing
    (0.36) would readmit off-topic questions under a semantic model whose
    unrelated-text similarities often exceed that. Run
    ``scripts/measure_gate.py --write`` to calibrate and re-enable it.
    """
    import os

    sig = "{}:{}".format(embedder.name, embedder.dimensions)
    row = db.one("SELECT * FROM calibrations WHERE embedder_signature = ?", (sig,))
    if row is not None:
        if "RELEVANCE_FLOOR" not in os.environ:
            settings.relevance_floor = float(row["relevance_floor"])
        if "DENSE_RESCUE_FLOOR" not in os.environ:
            settings.dense_rescue_floor = float(row["dense_rescue_floor"])
        return dict(row)
    if not embedder.name.startswith("hashing") and "DENSE_RESCUE_FLOOR" not in os.environ:
        settings.dense_rescue_floor = 1.0
    return None


def build_services(settings: Settings, embedder: Embedder, recover_jobs: bool = True) -> Services:
    settings.ensure_dirs()
    db = Database(settings.db_path)
    apply_calibration(settings, db, embedder)
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
        max_source_words=settings.max_source_words,
    )
    summary = import_legacy(settings.data_dir, db, registry, embedder)

    jobs = JobQueue(db, workers=settings.job_workers)

    def ingest(job: Job, ctx: JobContext) -> Dict:
        store = registry.store(job.notebook_id)  # type: ignore[arg-type]
        doc = store.process_source(
            job.payload["source_id"], cancelled=ctx.cancelled, progress=ctx.progress
        )
        if settings.source_guides:
            jobs.enqueue("guide", {"source_id": doc.doc_id}, notebook_id=job.notebook_id)
        return {"source_id": doc.doc_id, "chunk_count": doc.chunk_count}

    def guide(job: Job, ctx: JobContext) -> Dict:
        store = registry.store(job.notebook_id)  # type: ignore[arg-type]
        sid = job.payload["source_id"]
        src = store.get_source(sid)
        if src is None or src.status != "ready":
            return {"skipped": True}
        chunks = [{"id": c["id"], "text": c["text"]} for c in store.source_chunks(sid)]
        g = build_guide(chunks, src.title, get_llm(settings, "studio"))
        if src.guide.get("auto_label"):
            g["auto_label"] = src.guide["auto_label"]
        store.set_guide(sid, g)
        ctx.progress(1.0)
        ready = [s for s in store.list_sources() if s.status == "ready" and s.guide.get("summary")]
        if len(ready) >= AUTO_LABEL_MIN:
            jobs.enqueue("autolabel", {}, notebook_id=job.notebook_id)
        return {"source_id": sid, "generated_by": g["generated_by"]}

    def autolabel(job: Job, ctx: JobContext) -> Dict:
        store = registry.store(job.notebook_id)  # type: ignore[arg-type]
        ready = [s for s in store.list_sources() if s.status == "ready" and s.guide.get("summary")]
        if len(ready) < AUTO_LABEL_MIN:
            return {"skipped": True}
        labels = compute_labels(
            [{"id": s.id, "title": s.title, "topics": s.guide.get("topics", []),
              "summary": s.guide.get("summary", "")} for s in ready],
            get_llm(settings, "studio"))
        return {"updated": store.apply_auto_labels(labels)}

    jobs.register("ingest", ingest)
    jobs.register("guide", guide)
    jobs.register("autolabel", autolabel)
    if recover_jobs:
        jobs.recover()
    return Services(settings, db, registry, jobs, embedder, summary)
