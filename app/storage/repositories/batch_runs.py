"""Batch run repositories (runs and items)."""

from __future__ import annotations

import sqlite3

from app.storage.models import (
    BatchItemStatus,
    BatchRunItemRecord,
    BatchRunRecord,
    BatchRunStatus,
)
from app.storage.repositories.base import BaseRepository


class BatchRunRepository(BaseRepository):
    """CRUD operations for batch runs and their items."""

    # -- batch runs --------------------------------------------------------

    def upsert_run(self, batch: BatchRunRecord) -> str:
        return self._write(lambda connection: self._upsert_run_row(connection, batch))

    def list_runs(self, *, limit: int = 20) -> list[BatchRunRecord]:
        def _read(connection: sqlite3.Connection) -> list[BatchRunRecord]:
            rows = connection.execute(
                "SELECT * FROM batch_runs ORDER BY datetime(started_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._run_from_row(row) for row in rows]

        return self._read(_read)

    def get_run(self, batch_id: str) -> BatchRunRecord | None:
        def _read(connection: sqlite3.Connection) -> BatchRunRecord | None:
            row = connection.execute(
                "SELECT * FROM batch_runs WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            return self._run_from_row(row) if row else None

        return self._read(_read)

    # -- batch items -------------------------------------------------------

    def upsert_item(self, item: BatchRunItemRecord) -> str:
        return self._write(lambda connection: self._upsert_item_row(connection, item))

    def list_items(self, batch_id: str, *, limit: int = 200) -> list[BatchRunItemRecord]:
        def _read(connection: sqlite3.Connection) -> list[BatchRunItemRecord]:
            rows = connection.execute(
                "SELECT * FROM batch_run_items WHERE batch_id = ? ORDER BY uci_id ASC LIMIT ?",
                (batch_id, limit),
            ).fetchall()
            return [self._item_from_row(row) for row in rows]

        return self._read(_read)

    # -- row writers -------------------------------------------------------

    def _upsert_run_row(self, connection: sqlite3.Connection, batch: BatchRunRecord) -> str:
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
                self._dumps(batch.metadata),
                self._iso(batch.started_at),
                self._iso(batch.updated_at),
            ),
        )
        return batch.batch_id

    def _upsert_item_row(self, connection: sqlite3.Connection, item: BatchRunItemRecord) -> str:
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
                self._dumps(item.metadata),
                self._iso(item.created_at),
                self._iso(item.updated_at),
            ),
        )
        return item.item_id

    # -- row mappers -------------------------------------------------------

    def _run_from_row(self, row: sqlite3.Row) -> BatchRunRecord:
        return BatchRunRecord(
            batch_id=row["batch_id"],
            batch_name=row["batch_name"],
            total_datasets=row["total_datasets"],
            completed_count=row["completed_count"],
            failed_count=row["failed_count"],
            skipped_count=row["skipped_count"],
            status=BatchRunStatus(row["status"]),
            metadata=self._loads_dict(row["metadata_json"]),
            started_at=self._from_iso(row["started_at"]),
            updated_at=self._from_iso(row["updated_at"]),
        )

    def _item_from_row(self, row: sqlite3.Row) -> BatchRunItemRecord:
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
            metadata=self._loads_dict(row["metadata_json"]),
            created_at=self._from_iso(row["created_at"]),
            updated_at=self._from_iso(row["updated_at"]),
        )


__all__ = ["BatchRunRepository"]
