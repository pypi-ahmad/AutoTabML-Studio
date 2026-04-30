"""Job repository."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.storage.models import AppJobStatus, AppJobType, JobRecord
from app.storage.repositories.base import BaseRepository


class JobRepository(BaseRepository):
    """CRUD operations for job rows."""

    def record(self, job: JobRecord) -> str:
        return self._write(lambda connection: self._upsert_row(connection, job))

    def list_recent(
        self,
        *,
        limit: int = 20,
        job_type: AppJobType | None = None,
    ) -> list[JobRecord]:
        query = "SELECT * FROM jobs"
        params: list[object] = []
        if job_type is not None:
            query += " WHERE job_type = ?"
            params.append(job_type.value)
        query += " ORDER BY datetime(updated_at) DESC LIMIT ?"
        params.append(limit)

        def _read(connection: sqlite3.Connection) -> list[JobRecord]:
            rows = connection.execute(query, params).fetchall()
            return [self._from_row(row) for row in rows]

        return self._read(_read)

    def _upsert_row(self, connection: sqlite3.Connection, job: JobRecord) -> str:
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
                self._opt_str(job.primary_artifact_path),
                self._opt_str(job.summary_path),
                self._dumps(job.metadata),
                self._iso(job.created_at),
                self._iso(job.updated_at),
            ),
        )
        return job.job_id

    def _from_row(self, row: sqlite3.Row) -> JobRecord:
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
            metadata=self._loads_dict(row["metadata_json"]),
            created_at=self._from_iso(row["created_at"]),
            updated_at=self._from_iso(row["updated_at"]),
        )


JobRepo = JobRepository

__all__ = ["JobRepo", "JobRepository"]
