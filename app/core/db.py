"""SQLite access and the numbered-SQL migration runner.

One connection per ``Database``, guarded by a re-entrant lock. SQLite in WAL
mode lets readers proceed during a write. A single process does not need a
pool, and a single connection keeps transactions simple: ``with db.tx() as c``
either commits everything or nothing.

Migrations are ``app/migrations/NNN_name.sql`` files applied in numeric order
and recorded in ``schema_migrations``. A shipped migration is never edited;
changes go in a new file.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from array import array
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
_MIGRATION_NAME = re.compile(r"^(\d{3,})_[a-z0-9_]+\.sql$")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def pack_vector(vector: Sequence[float]) -> bytes:
    """float32 little-endian BLOB. Halves storage versus float64 JSON."""
    return array("f", vector).tobytes()


def unpack_vector(blob: bytes) -> List[float]:
    a = array("f")
    a.frombytes(blob)
    return a.tolist()


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> List[Tuple[int, Path]]:
    found = []
    for p in sorted(directory.glob("*.sql")):
        m = _MIGRATION_NAME.match(p.name)
        if not m:
            raise ValueError("Bad migration file name: {}".format(p.name))
        found.append((int(m.group(1)), p))
    numbers = [n for n, _ in found]
    if len(numbers) != len(set(numbers)):
        raise ValueError("Duplicate migration numbers in {}".format(directory))
    return sorted(found)


class Database:
    def __init__(self, path: Path, migrations_dir: Path = MIGRATIONS_DIR):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self.migrate(migrations_dir)

    # --- migrations ------------------------------------------------------

    def migrate(self, migrations_dir: Path = MIGRATIONS_DIR) -> List[int]:
        """Apply pending migrations. Returns the numbers applied."""
        applied: List[int] = []
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
            )
            done = {
                r[0] for r in self._conn.execute("SELECT version FROM schema_migrations")
            }
            for number, path in discover_migrations(migrations_dir):
                if number in done:
                    continue
                sql = path.read_text("utf-8")
                try:
                    self._conn.execute("BEGIN IMMEDIATE")
                    # executescript would COMMIT implicitly; run statements one
                    # by one so the migration and its record are atomic.
                    for stmt in _split_sql(sql):
                        self._conn.execute(stmt)
                    self._conn.execute(
                        "INSERT INTO schema_migrations VALUES (?, ?, ?)",
                        (number, path.name, utcnow()),
                    )
                    self._conn.execute("COMMIT")
                except Exception:
                    self._conn.execute("ROLLBACK")
                    logger.exception("Migration %s failed", path.name)
                    raise
                logger.info("Applied migration %s", path.name)
                applied.append(number)
        return applied

    @property
    def schema_version(self) -> int:
        row = self.one("SELECT MAX(version) AS v FROM schema_migrations")
        return int(row["v"] or 0)

    # --- queries ---------------------------------------------------------

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """A write transaction. Nested use joins the outer transaction."""
        with self._lock:
            if self._conn.in_transaction:
                yield self._conn
                return
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def execute(self, sql: str, params: Sequence = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def all(self, sql: str, params: Sequence = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: Sequence = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def get_meta(self, key: str) -> Optional[str]:
        row = self.one("SELECT value FROM app_meta WHERE key = ?", (key,))
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO app_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            (key, value),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _split_sql(sql: str) -> List[str]:
    """Split a migration into statements. Strips ``--`` comments; migrations
    must not put semicolons inside string literals or triggers."""
    lines = [ln.split("--", 1)[0] for ln in sql.splitlines()]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
