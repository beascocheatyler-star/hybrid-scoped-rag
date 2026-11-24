"""Artifact and study-progress persistence."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from app.core.db import Database, utcnow


def _row(r) -> Dict[str, Any]:
    d = dict(r)
    for k, default in (("params_json", {}), ("content_json", {}), ("source_ids_json", [])):
        try:
            d[k[:-5]] = json.loads(d.pop(k) or json.dumps(default))
        except ValueError:
            d[k[:-5]] = default
    return d


class ArtifactRepo:
    def __init__(self, db: Database):
        self.db = db

    def create(self, notebook_id: str, user_id: str, atype: str, params: Dict, title: str,
               selection: Optional[List[str]]) -> Dict:
        aid = uuid.uuid4().hex[:16]
        now = utcnow()
        p = dict(params)
        p["_selection"] = selection
        self.db.execute(
            "INSERT INTO artifacts(id, notebook_id, created_by, type, title, params_json, status, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?, 'queued', ?, ?)",
            (aid, notebook_id, user_id, atype, title[:200], json.dumps(p), now, now))
        return self.get(aid)  # type: ignore[return-value]

    def get(self, artifact_id: str) -> Optional[Dict]:
        r = self.db.one("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        return _row(r) if r else None

    def list(self, notebook_id: str) -> List[Dict]:
        return [_row(r) for r in self.db.all(
            "SELECT * FROM artifacts WHERE notebook_id = ? ORDER BY created_at DESC", (notebook_id,))]

    def set_job(self, artifact_id: str, job_id: str) -> None:
        self.db.execute("UPDATE artifacts SET job_id = ? WHERE id = ?", (job_id, artifact_id))

    def update(self, artifact_id: str, **fields) -> None:
        sets, params = ["updated_at = ?"], [utcnow()]
        for k, v in fields.items():
            col = k + "_json" if k in ("content", "source_ids", "params") else k
            sets.append("{} = ?".format(col))
            params.append(json.dumps(v, ensure_ascii=False) if col.endswith("_json") else v)
        self.db.execute("UPDATE artifacts SET {} WHERE id = ?".format(", ".join(sets)), params + [artifact_id])

    def delete(self, artifact_id: str) -> bool:
        return self.db.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,)).rowcount > 0

    # --- study progress (per user) --------------------------------------

    def progress(self, artifact_id: str, user_id: str) -> Dict[str, Dict]:
        return {r["item_id"]: {"state": r["state"], "attempts": r["attempts"], "answer": r["answer"]}
                for r in self.db.all("SELECT * FROM study_progress WHERE artifact_id = ? AND user_id = ?",
                                     (artifact_id, user_id))}

    def set_progress(self, artifact_id: str, user_id: str, item_id: str, state: str, answer: str = "") -> Dict:
        self.db.execute(
            "INSERT INTO study_progress(user_id, artifact_id, item_id, state, attempts, answer) "
            "VALUES (?,?,?,?,1,?) ON CONFLICT(user_id, artifact_id, item_id) DO UPDATE SET "
            "state = excluded.state, answer = excluded.answer, attempts = attempts + 1, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            (user_id, artifact_id, item_id, state, answer[:2000]))
        return self.progress(artifact_id, user_id)[item_id]

    def reset_progress(self, artifact_id: str, user_id: str, only: Optional[str] = None) -> None:
        if only:
            self.db.execute("DELETE FROM study_progress WHERE artifact_id = ? AND user_id = ? AND state = ?",
                            (artifact_id, user_id, only))
        else:
            self.db.execute("DELETE FROM study_progress WHERE artifact_id = ? AND user_id = ?",
                            (artifact_id, user_id))
