"""Project repository."""

from __future__ import annotations

import sqlite3

from app.storage.models import ProjectRecord
from app.storage.repositories.base import BaseRepository


class ProjectRepository(BaseRepository):
    """CRUD operations for project rows."""

    def upsert(self, project: ProjectRecord) -> None:
        self._write(lambda connection: self._upsert_row(connection, project))

    def upsert_in(self, connection: sqlite3.Connection, project: ProjectRecord) -> None:
        """Upsert using an existing connection (used during initial migration)."""

        self._upsert_row(connection, project)

    def get(self, project_id: str) -> ProjectRecord | None:
        def _read(connection: sqlite3.Connection) -> ProjectRecord | None:
            row = connection.execute(
                "SELECT * FROM projects WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            return self._from_row(row) if row else None

        return self._read(_read)

    def _upsert_row(self, connection: sqlite3.Connection, project: ProjectRecord) -> None:
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
                self._dumps(project.metadata),
                self._iso(project.created_at),
                self._iso(project.updated_at),
            ),
        )

    def _from_row(self, row: sqlite3.Row) -> ProjectRecord:
        return ProjectRecord(
            project_id=row["project_id"],
            name=row["name"],
            metadata=self._loads_dict(row["metadata_json"]),
            created_at=self._from_iso(row["created_at"]),
            updated_at=self._from_iso(row["updated_at"]),
        )


ProjectRepo = ProjectRepository

__all__ = ["ProjectRepo", "ProjectRepository"]
