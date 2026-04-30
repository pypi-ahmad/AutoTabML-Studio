"""Saved local model repository."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from app.storage.models import SavedLocalModelRecord
from app.storage.repositories.base import BaseRepository

if TYPE_CHECKING:
    from app.modeling.pycaret.schemas import SavedModelMetadata


class SavedModelRepository(BaseRepository):
    """CRUD operations for saved local model rows."""

    def upsert(self, saved_model: SavedLocalModelRecord) -> str:
        return self._write(lambda connection: self._upsert_row(connection, saved_model))

    def upsert_metadata(
        self,
        metadata: SavedModelMetadata,
        *,
        metadata_path: Path | None = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        record_id = str(metadata.model_path)
        return self.upsert(
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

    def list_recent(self, *, limit: int = 20) -> list[SavedLocalModelRecord]:
        def _read(connection: sqlite3.Connection) -> list[SavedLocalModelRecord]:
            rows = connection.execute(
                "SELECT * FROM saved_local_models ORDER BY datetime(updated_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._from_row(row) for row in rows]

        return self._read(_read)

    def _upsert_row(
        self,
        connection: sqlite3.Connection,
        saved_model: SavedLocalModelRecord,
    ) -> str:
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
                self._opt_str(saved_model.metadata_path),
                self._opt_str(saved_model.experiment_snapshot_path),
                self._dumps(saved_model.metadata),
                self._iso(saved_model.created_at),
                self._iso(saved_model.updated_at),
            ),
        )
        return saved_model.record_id

    def _from_row(self, row: sqlite3.Row) -> SavedLocalModelRecord:
        return SavedLocalModelRecord(
            record_id=row["record_id"],
            model_name=row["model_name"],
            model_path=Path(row["model_path"]),
            task_type=row["task_type"],
            target_column=row["target_column"],
            dataset_fingerprint=row["dataset_fingerprint"],
            metadata_path=Path(row["metadata_path"]) if row["metadata_path"] else None,
            experiment_snapshot_path=Path(row["experiment_snapshot_path"])
            if row["experiment_snapshot_path"]
            else None,
            metadata=self._loads_dict(row["metadata_json"]),
            created_at=self._from_iso(row["created_at"]),
            updated_at=self._from_iso(row["updated_at"]),
        )


__all__ = ["SavedModelRepository"]
