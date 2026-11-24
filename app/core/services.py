"""Wire the storage layer together: database, notebooks, legacy import, jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

from app.config import Settings
from app.core.db import Database
from app.core.embedding import Embedder
from app.core.guide import AUTO_LABEL_MIN, build_guide, compute_labels
from app.core.artifacts import ArtifactRepo
from app.core.importer import import_legacy
from app.core.studio import StudioRefused, generate as studio_generate
from app.core.overview import build_overview
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
        queue_overview(job.notebook_id)
        return {"source_id": sid, "generated_by": g["generated_by"]}

    def queue_overview(nb_id) -> None:
        """Debounced: at most one queued overview job per notebook."""
        if db.one("SELECT 1 FROM jobs WHERE notebook_id = ? AND type = 'overview' "
                  "AND status = 'queued'", (nb_id,)) is None:
            jobs.enqueue("overview", {}, notebook_id=nb_id)

    def overview(job: Job, ctx: JobContext) -> Dict:
        store = registry.store(job.notebook_id)  # type: ignore[arg-type]
        srcs = [{"id": s.id, "title": s.title, "guide": s.guide}
                for s in store.list_sources() if s.status == "ready"]
        ov = build_overview(srcs, get_llm(settings, "studio"))
        db.execute("UPDATE notebooks SET overview_json = ?, updated_at = updated_at WHERE id = ?",
                   (json.dumps(ov, ensure_ascii=False), job.notebook_id))
        return {"sources": len(srcs), "generated_by": ov["generated_by"]}

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

    arts = ArtifactRepo(db)

    def artifact(job: Job, ctx: JobContext) -> Dict:
        a = arts.get(job.payload["artifact_id"])
        if a is None:
            return {"skipped": True}
        arts.update(a["id"], status="running")
        store = registry.store(a["notebook_id"])
        params = {k: v for k, v in a["params"].items() if not k.startswith("_")}
        try:
            out = studio_generate(a["type"], params, store, a["params"].get("_selection"), settings,
                                  get_llm(settings, "studio"), registry.get(a["notebook_id"]).title,
                                  progress=ctx.progress, cancelled=ctx.cancelled)
        except StudioRefused as exc:
            arts.update(a["id"], status="failed", error=str(exc))
            raise
        except InterruptedError:
            arts.update(a["id"], status="cancelled", error="Cancelled.")
            raise
        except Exception as exc:
            arts.update(a["id"], status="failed", error=str(exc) or exc.__class__.__name__)
            raise
        if ctx.cancelled():
            arts.update(a["id"], status="cancelled", error="Cancelled.")
            raise InterruptedError()
        content = out["content"]
        content["_passage_chunk_ids"] = out["passage_chunk_ids"]
        sel = a["params"].get("_selection")
        ready = sorted(d.doc_id for d in store.list_documents())
        content["_snapshot_ids"] = ready if sel is None else sorted(x for x in sel if x in ready)
        arts.update(a["id"], status="ready", content=content, prompt_used=out["prompt_used"],
                    source_ids=out["source_ids"], generated_by=out["generated_by"], error="")
        return {"artifact_id": a["id"], "generated_by": out["generated_by"]}

    jobs.register("artifact", artifact)
    jobs.register("ingest", ingest)
    jobs.register("guide", guide)
    jobs.register("autolabel", autolabel)
    jobs.register("overview", overview)
    if recover_jobs:
        jobs.recover()
    return Services(settings, db, registry, jobs, embedder, summary)
