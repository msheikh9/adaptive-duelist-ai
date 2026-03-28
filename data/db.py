"""SQLite database connection and safe query execution."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "db"
DEFAULT_DB_PATH = DB_DIR / "adaptive_duelist.db"


class Database:
    """Thread-local SQLite database connection with safe query execution.

    All writes are wrapped in transactions. Errors during writes are
    logged and do not propagate — the game must never crash due to a
    database failure.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or DEFAULT_DB_PATH
        self._conn: sqlite3.Connection | None = None

    @property
    def path(self) -> Path:
        return self._path

    def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.connection.execute(sql, params)

    def executemany(self, sql: str, params_seq: list[tuple]) -> sqlite3.Cursor:
        return self.connection.executemany(sql, params_seq)

    def execute_safe(self, sql: str, params: tuple = ()) -> sqlite3.Cursor | None:
        """Execute a write query with error isolation.

        Returns the cursor on success, None on failure. Failures are
        logged but do not raise — the game loop must not crash from DB errors.
        """
        try:
            cursor = self.connection.execute(sql, params)
            self.connection.commit()
            return cursor
        except sqlite3.Error:
            logger.exception("Database write failed: %s", sql[:100])
            return None

    def executemany_safe(self, sql: str, params_seq: list[tuple]) -> bool:
        """Execute a batch write with error isolation. Returns True on success."""
        try:
            self.connection.executemany(sql, params_seq)
            self.connection.commit()
            return True
        except sqlite3.Error:
            logger.exception("Database batch write failed: %s", sql[:100])
            return False

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.connection.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.connection.execute(sql, params).fetchall()

    def table_exists(self, table_name: str) -> bool:
        row = self.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            (table_name,),
        )
        return row is not None

    def get_schema_version(self) -> int:
        if not self.table_exists("schema_version"):
            return 0
        row = self.fetchone(
            "SELECT MAX(version) as v FROM schema_version;"
        )
        if row is None or row["v"] is None:
            return 0
        return int(row["v"])
