"""SQLite-backed local app metadata store.

This store is a thin facade over the domain-scoped repositories defined in
:mod:`app.storage.repositories`. It owns the SQLite connector, runs migrations
once on first access, and forwards calls to the appropriate repository so
existing call sites keep working unchanged.

MLflow remains the source of truth for experiment/run tracking and registry
state; this store records local product-level metadata only.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from app.ingestion.schemas import LoadedDataset
from app.storage.migrations import apply_migrations
from app.storage.models import (
    AppJobType,
    BatchRunItemRecord,
    BatchRunRecord,
    DatasetRecord,
    JobRecord,
    ProjectRecord,
    SavedLocalModelRecord,
)
from app.storage.repositories import (
    BatchRunRepository,
    DatasetRepository,
    JobRepository,
    ProjectRepository,
    RepositoryContext,
    SavedModelRepository,
)
from app.storage.sqlite_connector import SQLiteConnector

if TYPE_CHECKING:
    from app.modeling.pycaret.schemas import SavedModelMetadata

_DEFAULT_PROJECT_ID = "local-workspace"
_DEFAULT_PROJECT_NAME = "Local Workspace"


class AppMetadataStore:
    """Repository facade for local workspace metadata.

    The store composes the per-domain repositories and exposes the original
    flat method surface used by the rest of the application. Tests that need to
    target a single domain (projects, datasets, jobs, saved models, batch runs)
    can also reach repositories directly via the public attributes.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._connector = SQLiteConnector(db_path)
        self._initialized = False
        self._initialization_lock = threading.RLock()

        context = RepositoryContext(
            connector=self._connector,
            initialize=self.initialize_if_needed,
        )
        self.projects = ProjectRepository(context)
        self.datasets = DatasetRepository(context)
        self.jobs = JobRepository(context)
        self.saved_models = SavedModelRepository(context)
        self.batch_runs = BatchRunRepository(context)

    @property
    def db_path(self) -> Path:
        return self._db_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        with self._initialization_lock:
            if self._initialized:
                return
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

            def _initialize(connection: sqlite3.Connection) -> None:
                apply_migrations(connection)
                self.projects.upsert_in(
                    connection,
                    ProjectRecord(project_id=_DEFAULT_PROJECT_ID, name=_DEFAULT_PROJECT_NAME),
                )

            self._connector.write(_initialize)
            self._initialized = True

    def initialize_if_needed(self) -> None:
        if self._initialized:
            return
        self.initialize()

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def upsert_project(self, project: ProjectRecord) -> None:
        self.projects.upsert(project)

    def get_project(self, project_id: str) -> ProjectRecord | None:
        return self.projects.get(project_id)

    def get_workspace_project(self) -> ProjectRecord:
        project = self.projects.get(_DEFAULT_PROJECT_ID)
        if project is not None:
            return project

        project = ProjectRecord(project_id=_DEFAULT_PROJECT_ID, name=_DEFAULT_PROJECT_NAME)
        self.projects.upsert(project)
        return project

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def upsert_dataset(self, dataset: DatasetRecord) -> str:
        return self.datasets.upsert(dataset)

    def upsert_dataset_from_loaded(
        self,
        loaded_dataset: LoadedDataset,
        *,
        dataset_name: str | None = None,
    ) -> str:
        return self.datasets.upsert_from_loaded(loaded_dataset, dataset_name=dataset_name)

    def list_recent_datasets(self, *, limit: int = 20) -> list[DatasetRecord]:
        return self.datasets.list_recent(limit=limit)

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def record_job(self, job: JobRecord) -> str:
        return self.jobs.record(job)

    def list_recent_jobs(
        self,
        *,
        limit: int = 20,
        job_type: AppJobType | None = None,
    ) -> list[JobRecord]:
        return self.jobs.list_recent(limit=limit, job_type=job_type)

    # ------------------------------------------------------------------
    # Saved local models
    # ------------------------------------------------------------------

    def upsert_saved_local_model(self, saved_model: SavedLocalModelRecord) -> str:
        return self.saved_models.upsert(saved_model)

    def upsert_saved_model_metadata(
        self,
        metadata: SavedModelMetadata,
        *,
        metadata_path: Path | None = None,
    ) -> str:
        return self.saved_models.upsert_metadata(metadata, metadata_path=metadata_path)

    def list_saved_local_models(self, *, limit: int = 20) -> list[SavedLocalModelRecord]:
        return self.saved_models.list_recent(limit=limit)

    # ------------------------------------------------------------------
    # Batch runs
    # ------------------------------------------------------------------

    def upsert_batch_run(self, batch: BatchRunRecord) -> str:
        return self.batch_runs.upsert_run(batch)

    def upsert_batch_item(self, item: BatchRunItemRecord) -> str:
        return self.batch_runs.upsert_item(item)

    def list_batch_runs(self, *, limit: int = 20) -> list[BatchRunRecord]:
        return self.batch_runs.list_runs(limit=limit)

    def get_batch_run(self, batch_id: str) -> BatchRunRecord | None:
        return self.batch_runs.get_run(batch_id)

    def list_batch_items(self, batch_id: str, *, limit: int = 200) -> list[BatchRunItemRecord]:
        return self.batch_runs.list_items(batch_id, limit=limit)
