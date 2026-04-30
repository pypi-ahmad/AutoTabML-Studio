"""Tests for the local SQLite metadata store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.config.models import AppSettings
from app.modeling.pycaret.schemas import ExperimentTaskType, SavedModelMetadata
from app.storage import AppJobStatus, AppJobType, AppMetadataStore, build_metadata_store
from app.storage.models import DatasetRecord, JobRecord
from app.storage.recorders import ensure_dataset_record
from app.storage.sqlite_connector import SQLiteConnector


def _create_legacy_database(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS app_metadata_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
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
            );
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
            );
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
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_type_created ON jobs(job_type, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_datasets_ingested_at ON datasets(ingested_at DESC);
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
            );
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
            );
            CREATE INDEX IF NOT EXISTS idx_batch_items_batch ON batch_run_items(batch_id, uci_id);
            """
        )
        connection.execute(
            "INSERT INTO app_metadata_info(key, value) VALUES('schema_version', '1')"
        )
        connection.execute(
            """
            INSERT INTO projects(project_id, name, metadata_json, created_at, updated_at)
            VALUES('legacy-project', 'Legacy Project', '{}', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
            """
        )
        connection.commit()


class TestMetadataStore:
    def test_build_metadata_store_returns_working_store(self, tmp_path: Path):
        settings = AppSettings(database={"path": tmp_path / "app.sqlite3"})

        store = build_metadata_store(settings)

        assert store is not None
        assert isinstance(store, AppMetadataStore)

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

    def test_initialize_enables_pragmas_and_versions_schema(self, tmp_path: Path):
        store = AppMetadataStore(tmp_path / "app.sqlite3")
        store.initialize()

        def _read_pragmas(connection: sqlite3.Connection) -> tuple[str, int, int, list[int]]:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
            foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
            versions = [
                row[0]
                for row in connection.execute(
                    "SELECT version FROM schema_migrations ORDER BY version ASC"
                ).fetchall()
            ]
            return journal_mode, synchronous, foreign_keys, versions

        journal_mode, synchronous, foreign_keys, versions = store._connector.read(_read_pragmas)

        assert journal_mode == "wal"
        assert synchronous == 1
        assert foreign_keys == 1
        assert versions == [1, 2]

    def test_initialize_migrates_legacy_database_in_place(self, tmp_path: Path):
        db_path = tmp_path / "legacy.sqlite3"
        _create_legacy_database(db_path)

        store = AppMetadataStore(db_path)
        store.initialize()

        with sqlite3.connect(db_path) as connection:
            versions = [
                row[0]
                for row in connection.execute(
                    "SELECT version FROM schema_migrations ORDER BY version ASC"
                ).fetchall()
            ]
            schema_version = connection.execute(
                "SELECT value FROM app_metadata_info WHERE key = 'schema_version'"
            ).fetchone()[0]
            legacy_project = connection.execute(
                "SELECT name FROM projects WHERE project_id = 'legacy-project'"
            ).fetchone()[0]

        assert versions == [1, 2]
        assert schema_version == "2"
        assert legacy_project == "Legacy Project"

    def test_foreign_keys_are_enforced_for_batch_items(self, tmp_path: Path):
        store = AppMetadataStore(tmp_path / "app.sqlite3")
        store.initialize()

        with pytest.raises(sqlite3.IntegrityError):
            store._connector.write(
                lambda connection: connection.execute(
                    """
                    INSERT INTO batch_run_items(
                        item_id, batch_id, uci_id, dataset_name, status, metadata_json, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "item-1",
                        "missing-batch",
                        1,
                        "iris",
                        "pending",
                        "{}",
                        "2026-01-01T00:00:00+00:00",
                        "2026-01-01T00:00:00+00:00",
                    ),
                )
            )

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


class TestSQLiteConnector:
    def test_write_retries_when_database_is_locked(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        connector = SQLiteConnector(tmp_path / "app.sqlite3", lock_retries=1, lock_backoff_seconds=0.01)
        attempts: list[str] = []
        sleeps: list[float] = []

        class _FakeConnection:
            def __init__(self, *, locked: bool = False) -> None:
                self.locked = locked
                self.rollback_calls = 0
                self.commit_calls = 0
                self.closed = False

            def execute(self, sql: str, *_args, **_kwargs):
                if sql == "BEGIN IMMEDIATE" and self.locked:
                    raise sqlite3.OperationalError("database is locked")
                attempts.append(sql)
                return self

            def commit(self) -> None:
                self.commit_calls += 1

            def rollback(self) -> None:
                self.rollback_calls += 1

            def close(self) -> None:
                self.closed = True

        first = _FakeConnection(locked=True)
        second = _FakeConnection(locked=False)
        fake_connections = iter([first, second])

        monkeypatch.setattr(connector, "_open_connection", lambda: next(fake_connections))
        monkeypatch.setattr("app.storage.sqlite_connector.time.sleep", sleeps.append)

        result = connector.write(lambda connection: connection.execute("SELECT 1") or "ok")

        assert result is second
        assert first.rollback_calls == 1
        assert second.commit_calls == 1
        assert first.closed and second.closed
        assert sleeps == [0.01]

    def test_write_rolls_back_atomically_on_exception(self, tmp_path: Path):
        connector = SQLiteConnector(tmp_path / "atomic.sqlite3")
        connector.write(
            lambda connection: connection.execute(
                "CREATE TABLE items (id TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
        )

        def _failing_insert(connection: sqlite3.Connection) -> None:
            connection.execute("INSERT INTO items(id, value) VALUES(?, ?)", ("item-1", "value"))
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            connector.write(_failing_insert)

        with connector.connect() as connection:
            row_count = connection.execute("SELECT COUNT(*) FROM items").fetchone()[0]

        assert row_count == 0