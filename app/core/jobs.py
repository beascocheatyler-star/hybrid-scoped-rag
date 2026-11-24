"""Background jobs: a ``jobs`` table plus an in-process thread pool.

A job row is the durable record; the thread pool is only the executor. Job
payloads point at data on disk (an ingest job carries a source id whose
original file is already stored), so any job can be re-run from its row
alone. On boot, ``recover()`` returns ``running`` jobs to ``queued`` and
resubmits every queued job. A crash mid-ingest therefore costs a retry, never
a lost upload.

Handlers are registered by type: ``queue.register("ingest", fn)`` where
``fn(job, ctx)`` may call ``ctx.progress(p)`` and should poll
``ctx.cancelled()`` between expensive steps.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Optional

from app.core.db import Database, utcnow

logger = logging.getLogger(__name__)

TERMINAL = ("succeeded", "failed", "cancelled")


@dataclass
class Job:
    id: str
    notebook_id: Optional[str]
    type: str
    payload: Dict
    status: str
    progress: float
    result: Dict
    error: str
    started_at: Optional[str]
    finished_at: Optional[str]
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_row(cls, r) -> "Job":
        return cls(
            id=r["id"], notebook_id=r["notebook_id"], type=r["type"],
            payload=json.loads(r["payload_json"] or "{}"), status=r["status"],
            progress=r["progress"], result=json.loads(r["result_json"] or "{}"),
            error=r["error"], started_at=r["started_at"], finished_at=r["finished_at"],
            created_at=r["created_at"], updated_at=r["updated_at"],
        )


@dataclass
class JobContext:
    job: Job
    _queue: "JobQueue" = field(repr=False)

    def progress(self, p: float) -> None:
        self._queue._set_progress(self.job.id, p)

    def cancelled(self) -> bool:
        return self._queue._is_cancel_requested(self.job.id)


class JobQueue:
    def __init__(self, db: Database, workers: int = 2):
        self.db = db
        self._handlers: Dict[str, Callable[[Job, JobContext], Optional[Dict]]] = {}
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers),
                                        thread_name_prefix="hsr-job")
        self._futures: Dict[str, Future] = {}
        self._cancel: set = set()
        self._lock = threading.RLock()
        self._closed = False

    def register(self, job_type: str, handler: Callable[[Job, JobContext], Optional[Dict]]) -> None:
        self._handlers[job_type] = handler

    # --- lifecycle -------------------------------------------------------

    def recover(self) -> int:
        """Re-queue interrupted work and resubmit all queued jobs."""
        self.db.execute(
            "UPDATE jobs SET status = 'queued', started_at = NULL, updated_at = ? "
            "WHERE status = 'running'",
            (utcnow(),),
        )
        self.db.execute(
            "UPDATE sources SET status = 'queued', updated_at = ? WHERE status = 'processing'",
            (utcnow(),),
        )
        ids = [r["id"] for r in self.db.all(
            "SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at")]
        for job_id in ids:
            self._submit(job_id)
        if ids:
            logger.info("Recovered %d queued job(s)", len(ids))
        return len(ids)

    def shutdown(self, wait: bool = True) -> None:
        self._closed = True
        self._pool.shutdown(wait=wait)

    # --- API -------------------------------------------------------------

    def enqueue(self, job_type: str, payload: Dict, notebook_id: Optional[str] = None,
                start: bool = True) -> Job:
        if job_type not in self._handlers:
            raise ValueError("No handler registered for job type {!r}".format(job_type))
        job_id = uuid.uuid4().hex[:16]
        now = utcnow()
        self.db.execute(
            "INSERT INTO jobs(id, notebook_id, type, payload_json, status, created_at, updated_at) "
            "VALUES (?,?,?,?, 'queued', ?, ?)",
            (job_id, notebook_id, job_type, json.dumps(payload), now, now),
        )
        if start:
            self._submit(job_id)
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> Optional[Job]:
        row = self.db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return Job.from_row(row) if row else None

    def cancel(self, job_id: str) -> Optional[Job]:
        job = self.get(job_id)
        if job is None or job.status in TERMINAL:
            return job
        with self._lock:
            self._cancel.add(job_id)
        # A queued job is cancelled immediately; a running one at its next
        # checkpoint.
        self.db.execute(
            "UPDATE jobs SET status = 'cancelled', finished_at = ?, updated_at = ? "
            "WHERE id = ? AND status = 'queued'",
            (utcnow(), utcnow(), job_id),
        )
        return self.get(job_id)

    def wait(self, job_id: str, timeout: Optional[float] = 30.0) -> Optional[Job]:
        """Block until the job finishes (tests, scripts)."""
        with self._lock:
            fut = self._futures.get(job_id)
        if fut is not None:
            try:
                fut.result(timeout=timeout)
            except Exception:
                pass
        return self.get(job_id)

    def drain(self, timeout: float = 60.0) -> None:
        with self._lock:
            futs = list(self._futures.values())
        for f in futs:
            try:
                f.result(timeout=timeout)
            except Exception:
                pass

    # --- internals -------------------------------------------------------

    def _submit(self, job_id: str) -> None:
        if self._closed:
            return
        with self._lock:
            existing = self._futures.get(job_id)
            if existing is not None and not existing.done():
                return
            self._futures[job_id] = self._pool.submit(self._run, job_id)

    def _is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancel

    def _set_progress(self, job_id: str, p: float) -> None:
        self.db.execute(
            "UPDATE jobs SET progress = ?, updated_at = ? WHERE id = ?",
            (max(0.0, min(1.0, float(p))), utcnow(), job_id),
        )

    def _finish(self, job_id: str, status: str, error: str = "",
                result: Optional[Dict] = None) -> None:
        self.db.execute(
            "UPDATE jobs SET status = ?, error = ?, result_json = ?, finished_at = ?, "
            "updated_at = ?, progress = CASE WHEN ? = 'succeeded' THEN 1 ELSE progress END "
            "WHERE id = ?",
            (status, error, json.dumps(result or {}), utcnow(), utcnow(), status, job_id),
        )

    def _run(self, job_id: str) -> None:
        # Claim atomically: only a queued job may start.
        cur = self.db.execute(
            "UPDATE jobs SET status = 'running', started_at = ?, updated_at = ? "
            "WHERE id = ? AND status = 'queued'",
            (utcnow(), utcnow(), job_id),
        )
        if cur.rowcount == 0:
            return
        job = self.get(job_id)
        handler = self._handlers.get(job.type) if job else None
        if job is None or handler is None:
            self._finish(job_id, "failed", "No handler for job type.")
            return
        ctx = JobContext(job, self)
        try:
            result = handler(job, ctx)
        except Exception as exc:
            from app.core.notebooks import JobCancelled

            if isinstance(exc, (JobCancelled, InterruptedError)) or ctx.cancelled():
                self._finish(job_id, "cancelled", "Cancelled.")
            else:
                logger.warning("Job %s (%s) failed: %s", job_id, job.type, exc)
                self._finish(job_id, "failed", str(exc) or exc.__class__.__name__)
            return
        finally:
            with self._lock:
                self._cancel.discard(job_id)
        self._finish(job_id, "succeeded", result=result or {})
