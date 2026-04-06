"""Tests for the local SQLite metadata store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.config.models import AppSettings
from app.modeling.pycaret.schemas import ExperimentTaskType, SavedModelMetadata
from app.storage import AppJobStatus, AppJobType, AppMetadataStore, build_metadata_store
from app.storage.models import DatasetRecord, JobRecord
from app.storage.recorders import ensure_dataset_record


class TestMetadataStore:
    def test_build_metadata_store_returns_none_without_database_config(self):
        settings = type("Settings", (), {})()

        assert build_metadata_store(settings) is None

    def test_ensure_dataset_record_tolerates_none_store(self):
        result = ensure_dataset_record(None, None, dataset_name="test")
        assert result is None

    def test_initialize_creates_default_project_and_lists_records(self, tmp_path: Path):
        store = AppMetadataStore(tmp_path / "app.sqlite3")
        store.initialize()

        dataset_record = DatasetRecord(
            dataset_key="dataset-1",
            project_id="local-workspace",
            display_name="train.csv",
            source_type="csv",
            source_locator="train.csv",
            schema_hash="schema-1",
            content_hash="content-1",
            row_count=10,
            column_count=3,
            column_names=["a", "b", "target"],
        )
        job_record = JobRecord(
            job_id="job-1",
            job_type=AppJobType.VALIDATION,
            status=AppJobStatus.SUCCESS,
            dataset_key="dataset-1",
            dataset_name="train.csv",
            title="Validation · train.csv",
        )

        store.upsert_dataset(dataset_record)
        store.record_job(job_record)

        with sqlite3.connect(store.db_path) as connection:
            project_row = connection.execute(
                "SELECT name FROM projects WHERE project_id = ?",
                ("local-workspace",),
            ).fetchone()

        recent_datasets = store.list_recent_datasets(limit=1)
        recent_jobs = store.list_recent_jobs(limit=1)
        workspace_project = store.get_workspace_project()

        assert project_row is not None
        assert project_row[0] == "Local Workspace"
        assert workspace_project.project_id == "local-workspace"
        assert recent_datasets[0].dataset_key == "dataset-1"
        assert recent_jobs[0].job_id == "job-1"

    def test_upsert_saved_model_metadata_lists_saved_models(self, tmp_path: Path):
        store = AppMetadataStore(tmp_path / "app.sqlite3")
        settings = AppSettings.model_validate({"artifacts": {"root_dir": str(tmp_path / "artifacts")}})

        model_path = settings.pycaret.models_dir / "smoke_model.pkl"
        metadata_path = settings.pycaret.artifacts_dir / "smoke_model_metadata.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"model")
        metadata_path.write_text("{}", encoding="utf-8")

        metadata = SavedModelMetadata(
            task_type=ExperimentTaskType.CLASSIFICATION,
            target_column="target",
            model_id="lr",
            model_name="SmokeModel",
            model_path=model_path,
            dataset_fingerprint="fp-1",
            feature_columns=["a", "b"],
            feature_dtypes={"a": "float64", "b": "int64"},
            target_dtype="int64",
        )

        store.upsert_saved_model_metadata(metadata, metadata_path=metadata_path)
        saved_models = store.list_saved_local_models(limit=1)

        assert saved_models[0].model_name == "SmokeModel"
        assert saved_models[0].model_path == model_path
        assert saved_models[0].metadata_path == metadata_path