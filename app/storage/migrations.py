"""Incremental SQLite schema migrations for the local metadata store."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

_VERSION_TABLE = "schema_migrations"
_LEGACY_INFO_TABLE = "app_metadata_info"


@dataclass(frozen=True)
class Migration:
    """A single ordered schema migration."""

    version: int
    name: str
    statements: tuple[str, ...]


_MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="core_metadata_schema",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS app_metadata_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_key TEXT PRIMARY KEY,
                project_id TEXT,
                display_name TEXT,
                source_type TEXT NOT NULL,
                source_locator TEXT NOT NULL,
                schema_hash TEXT NOT NULL,
                content_hash TEXT,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                column_names_json TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                ingested_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                project_id TEXT,
                dataset_key TEXT,
                dataset_name TEXT,
                title TEXT,
                mlflow_run_id TEXT,
                primary_artifact_path TEXT,
                summary_path TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS saved_local_models (
                record_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_path TEXT NOT NULL UNIQUE,
                task_type TEXT NOT NULL,
                target_column TEXT,
                dataset_fingerprint TEXT,
                metadata_path TEXT,
                experiment_snapshot_path TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_jobs_type_created ON jobs(job_type, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_datasets_ingested_at ON datasets(ingested_at DESC)",
        ),
    ),
    Migration(
        version=2,
        name="batch_run_schema",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS batch_runs (
                batch_id TEXT PRIMARY KEY,
                batch_name TEXT NOT NULL,
                total_datasets INTEGER NOT NULL DEFAULT 0,
                completed_count INTEGER NOT NULL DEFAULT 0,
                failed_count INTEGER NOT NULL DEFAULT 0,
                skipped_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'running',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS batch_run_items (
                item_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                uci_id INTEGER NOT NULL,
                dataset_name TEXT NOT NULL,
                target_column TEXT,
                task_type TEXT,
                row_count INTEGER,
                column_count INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                validation_status TEXT,
                profiling_status TEXT,
                benchmark_status TEXT,
                best_model TEXT,
                best_score REAL,
                ranking_metric TEXT,
                mlflow_run_id TEXT,
                duration_seconds REAL,
                error_message TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (batch_id) REFERENCES batch_runs(batch_id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_batch_items_batch ON batch_run_items(batch_id, uci_id)",
        ),
    ),
)

LATEST_SCHEMA_VERSION = _MIGRATIONS[-1].version


def apply_migrations(connection: sqlite3.Connection) -> int:
    """Create the version table and apply any pending schema migrations."""

    _ensure_version_table(connection)
    applied_versions = _get_applied_versions(connection)
    current_version = max(applied_versions, default=0)

    if not applied_versions:
        current_version = _detect_legacy_version(connection)
        if current_version > LATEST_SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {current_version} is newer than supported version {LATEST_SCHEMA_VERSION}."
            )
        _seed_legacy_history(connection, current_version)

    for migration in _MIGRATIONS:
        if migration.version <= current_version:
            continue
        for statement in migration.statements:
            connection.execute(statement)
        _record_migration(connection, migration.version, migration.name)
        _write_legacy_schema_version(connection, migration.version)
        current_version = migration.version

    return current_version


def _ensure_version_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_VERSION_TABLE} (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def _get_applied_versions(connection: sqlite3.Connection) -> set[int]:
    rows = connection.execute(f"SELECT version FROM {_VERSION_TABLE}").fetchall()
    return {int(row[0]) for row in rows}


def _detect_legacy_version(connection: sqlite3.Connection) -> int:
    if not _table_exists(connection, _LEGACY_INFO_TABLE):
        if _table_exists(connection, "batch_runs") and _table_exists(connection, "batch_run_items"):
            return 2
        if _table_exists(connection, "projects"):
            return 1
        return 0

    row = connection.execute(
        f"SELECT value FROM {_LEGACY_INFO_TABLE} WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        return 0
    try:
        return int(str(row[0]).strip())
    except (TypeError, ValueError):
        return 0


def _seed_legacy_history(connection: sqlite3.Connection, current_version: int) -> None:
    for migration in _MIGRATIONS:
        if migration.version > current_version:
            break
        _record_migration(connection, migration.version, migration.name)


def _record_migration(connection: sqlite3.Connection, version: int, name: str) -> None:
    connection.execute(
        f"INSERT OR IGNORE INTO {_VERSION_TABLE}(version, name, applied_at) VALUES(?, ?, ?)",
        (version, name, datetime.now(timezone.utc).isoformat()),
    )


def _write_legacy_schema_version(connection: sqlite3.Connection, version: int) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_LEGACY_INFO_TABLE} (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"INSERT OR REPLACE INTO {_LEGACY_INFO_TABLE}(key, value) VALUES('schema_version', ?)",
        (str(version),),
    )


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None