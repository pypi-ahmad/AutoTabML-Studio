"""SQLite-backed local app metadata store.

This store is intentionally small and local-first. It records product-level
metadata that helps the application navigate recent datasets, jobs, and saved
local models. MLflow remains the source of truth for experiment/run tracking
and registry state.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from app.ingestion.schemas import LoadedDataset
from app.storage.models import (
    AppJobStatus,
    AppJobType,
    BatchItemStatus,
    BatchRunItemRecord,
    BatchRunRecord,
    BatchRunStatus,
    DatasetRecord,
    JobRecord,
    ProjectRecord,
    SavedLocalModelRecord,
)

if TYPE_CHECKING:
    from app.modeling.pycaret.schemas import SavedModelMetadata

_SCHEMA_VERSION = "1"
_DEFAULT_PROJECT_ID = "local-workspace"
_DEFAULT_PROJECT_NAME = "Local Workspace"


class AppMetadataStore:
    """Minimal repository for local workspace metadata."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @property
    def db_path(self) -> Path:
        return self._db_path

    def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS app_metadata_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
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
                """
            )
            cursor.execute(
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
                """
            )
            cursor.execute(
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
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_type_created ON jobs(job_type, created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_ingested_at ON datasets(ingested_at DESC)")
            cursor.execute(
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
                """
            )
            cursor.execute(
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
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_items_batch ON batch_run_items(batch_id, uci_id)")
            cursor.execute(
                "INSERT OR REPLACE INTO app_metadata_info(key, value) VALUES('schema_version', ?) ",
                (_SCHEMA_VERSION,),
            )
            connection.commit()

        self.upsert_project(ProjectRecord(project_id=_DEFAULT_PROJECT_ID, name=_DEFAULT_PROJECT_NAME))

    def upsert_project(self, project: ProjectRecord) -> None:
        self.initialize_if_needed()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO projects(project_id, name, metadata_json, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    name=excluded.name,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    project.project_id,
                    project.name,
                    json.dumps(project.metadata, default=str),
                    project.created_at.isoformat(),
                    project.updated_at.isoformat(),
                ),
            )
            connection.commit()

    def get_project(self, project_id: str) -> ProjectRecord | None:
        self.initialize_if_needed()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM projects WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        if row is None:
            return None
        return self._project_from_row(row)

    def get_workspace_project(self) -> ProjectRecord:
        project = self.get_project(_DEFAULT_PROJECT_ID)
        if project is not None:
            return project

        project = ProjectRecord(project_id=_DEFAULT_PROJECT_ID, name=_DEFAULT_PROJECT_NAME)
        self.upsert_project(project)
        return project

    def upsert_dataset(self, dataset: DatasetRecord) -> str:
        self.initialize_if_needed()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO datasets(
                    dataset_key, project_id, display_name, source_type, source_locator, schema_hash,
                    content_hash, row_count, column_count, column_names_json, tags_json, metadata_json, ingested_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_key) DO UPDATE SET
                    project_id=excluded.project_id,
                    display_name=excluded.display_name,
                    source_type=excluded.source_type,
                    source_locator=excluded.source_locator,
                    schema_hash=excluded.schema_hash,
                    content_hash=excluded.content_hash,
                    row_count=excluded.row_count,
                    column_count=excluded.column_count,
                    column_names_json=excluded.column_names_json,
                    tags_json=excluded.tags_json,
                    metadata_json=excluded.metadata_json,
                    ingested_at=excluded.ingested_at
                """,
                (
                    dataset.dataset_key,
                    dataset.project_id,
                    dataset.display_name,
                    dataset.source_type,
                    dataset.source_locator,
                    dataset.schema_hash,
                    dataset.content_hash,
                    dataset.row_count,
                    dataset.column_count,
                    json.dumps(dataset.column_names),
                    json.dumps(dataset.tags),
                    json.dumps(dataset.metadata, default=str),
                    dataset.ingested_at.isoformat(),
                ),
            )
            connection.commit()
        return dataset.dataset_key

    def upsert_dataset_from_loaded(self, loaded_dataset: LoadedDataset, *, dataset_name: str | None = None) -> str:
        metadata = loaded_dataset.metadata
        dataset_key = metadata.content_hash or metadata.schema_hash or metadata.source_locator
        record = DatasetRecord(
            dataset_key=str(dataset_key),
            project_id=metadata.project_id or _DEFAULT_PROJECT_ID,
            display_name=dataset_name or metadata.display_name,
            source_type=metadata.source_type.value,
            source_locator=metadata.source_locator,
            schema_hash=metadata.schema_hash,
            content_hash=metadata.content_hash,
            row_count=metadata.row_count,
            column_count=metadata.column_count,
            column_names=list(metadata.column_names),
            tags=list(metadata.tags),
            metadata=metadata.model_dump(mode="json"),
            ingested_at=metadata.ingestion_timestamp,
        )
        return self.upsert_dataset(record)

    def record_job(self, job: JobRecord) -> str:
        self.initialize_if_needed()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs(
                    job_id, job_type, status, project_id, dataset_key, dataset_name, title,
                    mlflow_run_id, primary_artifact_path, summary_path, metadata_json, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    job_type=excluded.job_type,
                    status=excluded.status,
                    project_id=excluded.project_id,
                    dataset_key=excluded.dataset_key,
                    dataset_name=excluded.dataset_name,
                    title=excluded.title,
                    mlflow_run_id=excluded.mlflow_run_id,
                    primary_artifact_path=excluded.primary_artifact_path,
                    summary_path=excluded.summary_path,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    job.job_id,
                    job.job_type.value,
                    job.status.value,
                    job.project_id,
                    job.dataset_key,
                    job.dataset_name,
                    job.title,
                    job.mlflow_run_id,
                    str(job.primary_artifact_path) if job.primary_artifact_path is not None else None,
                    str(job.summary_path) if job.summary_path is not None else None,
                    json.dumps(job.metadata, default=str),
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )
            connection.commit()
        return job.job_id

    def upsert_saved_local_model(
        self,
        saved_model: SavedLocalModelRecord,
    ) -> str:
        self.initialize_if_needed()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO saved_local_models(
                    record_id, model_name, model_path, task_type, target_column, dataset_fingerprint,
                    metadata_path, experiment_snapshot_path, metadata_json, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_path) DO UPDATE SET
                    record_id=excluded.record_id,
                    model_name=excluded.model_name,
                    task_type=excluded.task_type,
                    target_column=excluded.target_column,
                    dataset_fingerprint=excluded.dataset_fingerprint,
                    metadata_path=excluded.metadata_path,
                    experiment_snapshot_path=excluded.experiment_snapshot_path,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    saved_model.record_id,
                    saved_model.model_name,
                    str(saved_model.model_path),
                    saved_model.task_type,
                    saved_model.target_column,
                    saved_model.dataset_fingerprint,
                    str(saved_model.metadata_path) if saved_model.metadata_path is not None else None,
                    str(saved_model.experiment_snapshot_path) if saved_model.experiment_snapshot_path is not None else None,
                    json.dumps(saved_model.metadata, default=str),
                    saved_model.created_at.isoformat(),
                    saved_model.updated_at.isoformat(),
                ),
            )
            connection.commit()
        return saved_model.record_id

    def upsert_saved_model_metadata(
        self,
        metadata: SavedModelMetadata,
        *,
        metadata_path: Path | None = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        record_id = str(metadata.model_path)
        return self.upsert_saved_local_model(
            SavedLocalModelRecord(
                record_id=record_id,
                model_name=metadata.model_name,
                model_path=metadata.model_path,
                task_type=metadata.task_type.value,
                target_column=metadata.target_column,
                dataset_fingerprint=metadata.dataset_fingerprint,
                metadata_path=metadata_path,
                experiment_snapshot_path=metadata.experiment_snapshot_path,
                metadata=metadata.model_dump(mode="json"),
                created_at=now,
                updated_at=now,
            )
        )

    def list_recent_jobs(self, *, limit: int = 20, job_type: AppJobType | None = None) -> list[JobRecord]:
        self.initialize_if_needed()
        query = "SELECT * FROM jobs"
        params: list[object] = []
        if job_type is not None:
            query += " WHERE job_type = ?"
            params.append(job_type.value)
        query += " ORDER BY datetime(updated_at) DESC LIMIT ?"
        params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._job_from_row(row) for row in rows]

    def list_recent_datasets(self, *, limit: int = 20) -> list[DatasetRecord]:
        self.initialize_if_needed()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM datasets ORDER BY datetime(ingested_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._dataset_from_row(row) for row in rows]

    def list_saved_local_models(self, *, limit: int = 20) -> list[SavedLocalModelRecord]:
        self.initialize_if_needed()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM saved_local_models ORDER BY datetime(updated_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._saved_model_from_row(row) for row in rows]

    # ------------------------------------------------------------------
    # Batch run CRUD
    # ------------------------------------------------------------------

    def upsert_batch_run(self, batch: BatchRunRecord) -> str:
        self.initialize_if_needed()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO batch_runs(
                    batch_id, batch_name, total_datasets, completed_count, failed_count,
                    skipped_count, status, metadata_json, started_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    batch_name=excluded.batch_name,
                    total_datasets=excluded.total_datasets,
                    completed_count=excluded.completed_count,
                    failed_count=excluded.failed_count,
                    skipped_count=excluded.skipped_count,
                    status=excluded.status,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    batch.batch_id,
                    batch.batch_name,
                    batch.total_datasets,
                    batch.completed_count,
                    batch.failed_count,
                    batch.skipped_count,
                    batch.status.value,
                    json.dumps(batch.metadata, default=str),
                    batch.started_at.isoformat(),
                    batch.updated_at.isoformat(),
                ),
            )
            connection.commit()
        return batch.batch_id

    def upsert_batch_item(self, item: BatchRunItemRecord) -> str:
        self.initialize_if_needed()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO batch_run_items(
                    item_id, batch_id, uci_id, dataset_name, target_column, task_type,
                    row_count, column_count, status, validation_status, profiling_status,
                    benchmark_status, best_model, best_score, ranking_metric, mlflow_run_id,
                    duration_seconds, error_message, metadata_json, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    status=excluded.status,
                    target_column=excluded.target_column,
                    task_type=excluded.task_type,
                    row_count=excluded.row_count,
                    column_count=excluded.column_count,
                    validation_status=excluded.validation_status,
                    profiling_status=excluded.profiling_status,
                    benchmark_status=excluded.benchmark_status,
                    best_model=excluded.best_model,
                    best_score=excluded.best_score,
                    ranking_metric=excluded.ranking_metric,
                    mlflow_run_id=excluded.mlflow_run_id,
                    duration_seconds=excluded.duration_seconds,
                    error_message=excluded.error_message,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    item.item_id,
                    item.batch_id,
                    item.uci_id,
                    item.dataset_name,
                    item.target_column,
                    item.task_type,
                    item.row_count,
                    item.column_count,
                    item.status.value,
                    item.validation_status,
                    item.profiling_status,
                    item.benchmark_status,
                    item.best_model,
                    item.best_score,
                    item.ranking_metric,
                    item.mlflow_run_id,
                    item.duration_seconds,
                    item.error_message,
                    json.dumps(item.metadata, default=str),
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                ),
            )
            connection.commit()
        return item.item_id

    def list_batch_runs(self, *, limit: int = 20) -> list[BatchRunRecord]:
        self.initialize_if_needed()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM batch_runs ORDER BY datetime(started_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._batch_run_from_row(row) for row in rows]

    def get_batch_run(self, batch_id: str) -> BatchRunRecord | None:
        self.initialize_if_needed()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM batch_runs WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        return self._batch_run_from_row(row) if row else None

    def list_batch_items(self, batch_id: str, *, limit: int = 200) -> list[BatchRunItemRecord]:
        self.initialize_if_needed()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM batch_run_items WHERE batch_id = ? ORDER BY uci_id ASC LIMIT ?",
                (batch_id, limit),
            ).fetchall()
        return [self._batch_item_from_row(row) for row in rows]

    def initialize_if_needed(self) -> None:
        if self._db_path.exists():
            return
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _dataset_from_row(self, row: sqlite3.Row) -> DatasetRecord:
        return DatasetRecord(
            dataset_key=row["dataset_key"],
            project_id=row["project_id"],
            display_name=row["display_name"],
            source_type=row["source_type"],
            source_locator=row["source_locator"],
            schema_hash=row["schema_hash"],
            content_hash=row["content_hash"],
            row_count=row["row_count"],
            column_count=row["column_count"],
            column_names=json.loads(row["column_names_json"] or "[]"),
            tags=json.loads(row["tags_json"] or "[]"),
            metadata=json.loads(row["metadata_json"] or "{}"),
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
        )

    def _project_from_row(self, row: sqlite3.Row) -> ProjectRecord:
        return ProjectRecord(
            project_id=row["project_id"],
            name=row["name"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _job_from_row(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            job_type=AppJobType(row["job_type"]),
            status=AppJobStatus(row["status"]),
            project_id=row["project_id"],
            dataset_key=row["dataset_key"],
            dataset_name=row["dataset_name"],
            title=row["title"],
            mlflow_run_id=row["mlflow_run_id"],
            primary_artifact_path=Path(row["primary_artifact_path"]) if row["primary_artifact_path"] else None,
            summary_path=Path(row["summary_path"]) if row["summary_path"] else None,
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _saved_model_from_row(self, row: sqlite3.Row) -> SavedLocalModelRecord:
        return SavedLocalModelRecord(
            record_id=row["record_id"],
            model_name=row["model_name"],
            model_path=Path(row["model_path"]),
            task_type=row["task_type"],
            target_column=row["target_column"],
            dataset_fingerprint=row["dataset_fingerprint"],
            metadata_path=Path(row["metadata_path"]) if row["metadata_path"] else None,
            experiment_snapshot_path=Path(row["experiment_snapshot_path"]) if row["experiment_snapshot_path"] else None,
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _batch_run_from_row(self, row: sqlite3.Row) -> BatchRunRecord:
        return BatchRunRecord(
            batch_id=row["batch_id"],
            batch_name=row["batch_name"],
            total_datasets=row["total_datasets"],
            completed_count=row["completed_count"],
            failed_count=row["failed_count"],
            skipped_count=row["skipped_count"],
            status=BatchRunStatus(row["status"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
            started_at=datetime.fromisoformat(row["started_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _batch_item_from_row(self, row: sqlite3.Row) -> BatchRunItemRecord:
        return BatchRunItemRecord(
            item_id=row["item_id"],
            batch_id=row["batch_id"],
            uci_id=row["uci_id"],
            dataset_name=row["dataset_name"],
            target_column=row["target_column"],
            task_type=row["task_type"],
            row_count=row["row_count"],
            column_count=row["column_count"],
            status=BatchItemStatus(row["status"]),
            validation_status=row["validation_status"],
            profiling_status=row["profiling_status"],
            benchmark_status=row["benchmark_status"],
            best_model=row["best_model"],
            best_score=row["best_score"],
            ranking_metric=row["ranking_metric"],
            mlflow_run_id=row["mlflow_run_id"],
            duration_seconds=row["duration_seconds"],
            error_message=row["error_message"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )