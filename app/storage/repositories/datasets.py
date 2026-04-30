"""Dataset repository."""

from __future__ import annotations

import sqlite3

from app.ingestion.schemas import LoadedDataset
from app.storage.models import DatasetRecord
from app.storage.repositories.base import BaseRepository

_DEFAULT_PROJECT_ID = "local-workspace"


class DatasetRepository(BaseRepository):
    """CRUD operations for dataset rows."""

    def upsert(self, dataset: DatasetRecord) -> str:
        return self._write(lambda connection: self._upsert_row(connection, dataset))

    def upsert_from_loaded(
        self,
        loaded_dataset: LoadedDataset,
        *,
        dataset_name: str | None = None,
    ) -> str:
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
        return self.upsert(record)

    def list_recent(self, *, limit: int = 20) -> list[DatasetRecord]:
        def _read(connection: sqlite3.Connection) -> list[DatasetRecord]:
            rows = connection.execute(
                "SELECT * FROM datasets ORDER BY datetime(ingested_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._from_row(row) for row in rows]

        return self._read(_read)

    def _upsert_row(self, connection: sqlite3.Connection, dataset: DatasetRecord) -> str:
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
                self._dumps(dataset.column_names),
                self._dumps(dataset.tags),
                self._dumps(dataset.metadata),
                self._iso(dataset.ingested_at),
            ),
        )
        return dataset.dataset_key

    def _from_row(self, row: sqlite3.Row) -> DatasetRecord:
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
            column_names=self._loads_list(row["column_names_json"]),
            tags=self._loads_list(row["tags_json"]),
            metadata=self._loads_dict(row["metadata_json"]),
            ingested_at=self._from_iso(row["ingested_at"]),
        )


DatasetRepo = DatasetRepository

__all__ = ["DatasetRepo", "DatasetRepository"]
