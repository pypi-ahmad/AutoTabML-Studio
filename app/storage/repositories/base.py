"""Shared base utilities for storage repositories."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from app.storage.sqlite_connector import SQLiteConnector

T = TypeVar("T")


@dataclass
class RepositoryContext:
    """Bundle of shared dependencies passed to every repository.

    The context holds the SQLite connector plus a lazy-initialization callback
    invoked the first time any repository performs a read or write. The callback
    is typically the metadata store's migration runner.
    """

    connector: SQLiteConnector
    initialize: Callable[[], None]


class BaseRepository:
    """Common access patterns shared by every domain repository.

    Subclasses focus on SQL + row mapping; connection management, write
    transactions, lazy migration, and JSON helpers live here.
    """

    def __init__(self, context: RepositoryContext) -> None:
        self._context = context

    # -- shared connector access -------------------------------------------------

    def _connect(self):
        return self._context.connector.connect()

    def _read(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        self._context.initialize()
        return self._context.connector.read(operation)

    def _write(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        self._context.initialize()
        return self._context.connector.write(operation)

    # -- shared serialization helpers -------------------------------------------

    @staticmethod
    def _dumps(value: object) -> str:
        return json.dumps(value, default=str)

    @staticmethod
    def _loads_dict(raw: str | None) -> dict:
        return json.loads(raw or "{}")

    @staticmethod
    def _loads_list(raw: str | None) -> list:
        return json.loads(raw or "[]")

    @staticmethod
    def _iso(value: datetime) -> str:
        return value.isoformat()

    @staticmethod
    def _from_iso(value: str) -> datetime:
        return datetime.fromisoformat(value)

    @staticmethod
    def _opt_path(value: object) -> Path | None:
        return Path(value) if value else None

    @staticmethod
    def _opt_str(value: object) -> str | None:
        return str(value) if value is not None else None


__all__ = ["BaseRepository", "RepositoryContext"]
