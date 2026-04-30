"""Domain repository tests for the modular storage layer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.storage.models import (
    AppJobStatus,
    AppJobType,
    BatchItemStatus,
    BatchRunItemRecord,
    BatchRunRecord,
    BatchRunStatus,
    DatasetRecord,
    JobRecord,
    SavedLocalModelRecord,
)
from app.storage.repositories import (
    BatchRunRepository,
    DatasetRepo,
    DatasetRepository,
    JobRepo,
    JobRepository,
    ProjectRepo,
    ProjectRepository,
    RepositoryContext,
    SavedModelRepository,
)
from app.storage.sqlite_connector import SQLiteConnector
from app.storage.store import AppMetadataStore


def _now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture()
def store(tmp_path: Path) -> AppMetadataStore:
    store = AppMetadataStore(tmp_path / "app.sqlite3")
    store.initialize_if_needed()
    return store


def _dataset(key: str = "ds-1") -> DatasetRecord:
    return DatasetRecord(
        dataset_key=key,
        project_id="local-workspace",
        display_name="Sample",
        source_type="csv",
        source_locator="/tmp/sample.csv",
        schema_hash="schema",
        content_hash="content",
        row_count=10,
        column_count=2,
        column_names=["a", "b"],
        tags=["x"],
        metadata={"k": "v"},
        ingested_at=_now(),
    )


def _job(job_id: str = "job-1") -> JobRecord:
    return JobRecord(
        job_id=job_id,
        job_type=AppJobType.BENCHMARK,
        status=AppJobStatus.SUCCESS,
        project_id="local-workspace",
        dataset_key="ds-1",
        dataset_name="Sample",
        title="run",
        mlflow_run_id=None,
        primary_artifact_path=None,
        summary_path=None,
        metadata={},
        created_at=_now(),
        updated_at=_now(),
    )


class TestStoreFacade:
    def test_repositories_are_exposed(self, store: AppMetadataStore) -> None:
        assert isinstance(store.projects, ProjectRepository)
        assert isinstance(store.projects, ProjectRepo)
        assert isinstance(store.datasets, DatasetRepository)
        assert isinstance(store.datasets, DatasetRepo)
        assert isinstance(store.jobs, JobRepository)
        assert isinstance(store.jobs, JobRepo)
        assert isinstance(store.saved_models, SavedModelRepository)
        assert isinstance(store.batch_runs, BatchRunRepository)

    def test_facade_delegates_to_repositories(self, store: AppMetadataStore) -> None:
        store.upsert_dataset(_dataset())
        store.record_job(_job())
        assert [d.dataset_key for d in store.list_recent_datasets()] == ["ds-1"]
        assert [j.job_id for j in store.list_recent_jobs()] == ["job-1"]


class TestProjectRepository:
    def test_get_workspace_project_creates_default(self, store: AppMetadataStore) -> None:
        project = store.get_workspace_project()
        assert project.project_id == "local-workspace"
        assert store.projects.get("local-workspace") is not None

    def test_get_unknown_returns_none(self, store: AppMetadataStore) -> None:
        assert store.projects.get("missing") is None


class TestDatasetRepository:
    def test_upsert_and_list(self, store: AppMetadataStore) -> None:
        store.datasets.upsert(_dataset("a"))
        store.datasets.upsert(_dataset("b"))
        keys = [d.dataset_key for d in store.datasets.list_recent()]
        assert set(keys) == {"a", "b"}

    def test_upsert_updates_existing_row(self, store: AppMetadataStore) -> None:
        record = _dataset("a")
        store.datasets.upsert(record)
        record.display_name = "Renamed"
        store.datasets.upsert(record)
        records = store.datasets.list_recent()
        assert len(records) == 1
        assert records[0].display_name == "Renamed"


class TestJobRepository:
    def test_filter_by_job_type(self, store: AppMetadataStore) -> None:
        store.jobs.record(_job("benchmark-1"))
        validation = _job("validation-1")
        validation.job_type = AppJobType.VALIDATION
        store.jobs.record(validation)

        benchmarks = store.jobs.list_recent(job_type=AppJobType.BENCHMARK)
        assert [j.job_id for j in benchmarks] == ["benchmark-1"]


class TestBatchRunRepository:
    def test_round_trip_run_and_items(self, store: AppMetadataStore) -> None:
        run = BatchRunRecord(
            batch_id="batch-1",
            batch_name="Demo",
            total_datasets=1,
            completed_count=0,
            failed_count=0,
            skipped_count=0,
            status=BatchRunStatus.RUNNING,
            metadata={},
            started_at=_now(),
            updated_at=_now(),
        )
        store.batch_runs.upsert_run(run)

        item = BatchRunItemRecord(
            item_id="item-1",
            batch_id="batch-1",
            uci_id=1,
            dataset_name="Demo dataset",
            target_column="y",
            task_type="classification",
            row_count=100,
            column_count=5,
            status=BatchItemStatus.SUCCESS,
            validation_status="ok",
            profiling_status="ok",
            benchmark_status="ok",
            best_model="rf",
            best_score=0.91,
            ranking_metric="accuracy",
            mlflow_run_id=None,
            duration_seconds=1.5,
            error_message=None,
            metadata={},
            created_at=_now(),
            updated_at=_now(),
        )
        store.batch_runs.upsert_item(item)

        assert store.batch_runs.get_run("batch-1") is not None
        items = store.batch_runs.list_items("batch-1")
        assert [i.item_id for i in items] == ["item-1"]


class TestSavedModelRepository:
    def test_upsert_and_list(self, store: AppMetadataStore) -> None:
        record = SavedLocalModelRecord(
            record_id="m-1",
            model_name="rf",
            model_path=Path("models/rf.pkl"),
            task_type="classification",
            target_column="y",
            dataset_fingerprint="fp",
            metadata_path=None,
            experiment_snapshot_path=None,
            metadata={},
            created_at=_now(),
            updated_at=_now(),
        )
        store.saved_models.upsert(record)
        records = store.saved_models.list_recent()
        assert [r.record_id for r in records] == ["m-1"]


class TestRepositoryContextSharing:
    def test_repositories_share_connector_and_lazy_init(self, tmp_path: Path) -> None:
        connector = SQLiteConnector(tmp_path / "shared.sqlite3")
        calls = {"count": 0}

        def init() -> None:
            calls["count"] += 1
            # First call applies migrations via the store; subsequent calls are no-ops.
            store.initialize_if_needed()

        store = AppMetadataStore(tmp_path / "shared.sqlite3")
        # Ensure repositories share a single context with the store's connector.
        assert store.projects._context.connector is store._connector
        assert store.datasets._context.connector is store._connector

        # Build an independent context bound to the same connector to confirm isolation.
        independent = ProjectRepository(
            RepositoryContext(connector=connector, initialize=init)
        )
        store.initialize_if_needed()
        # Reading a missing project should not raise even though `init` is custom.
        assert independent.get("missing") is None
        assert calls["count"] >= 1
