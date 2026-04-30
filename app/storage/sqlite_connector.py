"""Reusable SQLite connector with safe defaults for local metadata storage."""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from app.errors import log_exception

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_TIMEOUT_SECONDS = 5.0
_DEFAULT_BUSY_TIMEOUT_MS = 5_000
_DEFAULT_LOCK_RETRIES = 4
_DEFAULT_LOCK_BACKOFF_SECONDS = 0.05
_LOCKED_ERROR_MARKERS = (
    "database is locked",
    "database table is locked",
    "database schema is locked",
)


class SQLiteConnector:
    """Open SQLite connections with consistent PRAGMAs and atomic write helpers."""

    def __init__(
        self,
        db_path: Path,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
        lock_retries: int = _DEFAULT_LOCK_RETRIES,
        lock_backoff_seconds: float = _DEFAULT_LOCK_BACKOFF_SECONDS,
    ) -> None:
        self._db_path = db_path
        self._timeout_seconds = timeout_seconds
        self._busy_timeout_ms = busy_timeout_ms
        self._lock_retries = lock_retries
        self._lock_backoff_seconds = lock_backoff_seconds

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection for read operations."""

        connection = self._open_connection()
        try:
            yield connection
        finally:
            connection.close()

    def read(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        """Run a read-only callback using a configured connection."""

        with self.connect() as connection:
            return operation(connection)

    def write(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        """Run a write callback in an atomic transaction with lock retries."""

        delay = self._lock_backoff_seconds
        for attempt in range(self._lock_retries + 1):
            with self.connect() as connection:
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    result = operation(connection)
                    connection.commit()
                    return result
                except sqlite3.OperationalError as exc:
                    connection.rollback()
                    if self._is_locked_error(exc) and attempt < self._lock_retries:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    log_exception(
                        logger,
                        exc,
                        operation="sqlite.write",
                        context={"db_path": str(self._db_path), "attempt": attempt},
                    )
                    raise
                except sqlite3.DatabaseError as exc:
                    connection.rollback()
                    log_exception(
                        logger,
                        exc,
                        operation="sqlite.write",
                        context={"db_path": str(self._db_path), "attempt": attempt},
                    )
                    raise

        raise RuntimeError("SQLite write retry loop exhausted unexpectedly.")

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._db_path,
            timeout=self._timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        return connection

    @staticmethod
    def _is_locked_error(exc: sqlite3.OperationalError) -> bool:
        message = str(exc).lower()
        return any(marker in message for marker in _LOCKED_ERROR_MARKERS)