"""Lightweight smoke coverage for the local-first runtime hardening flow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config.enums import ExecutionBackend, WorkspaceMode
from app.config.models import AppSettings, ProfilingMode
from app.ingestion import DatasetInputSpec, IngestionSourceType, load_dataset
from app.modeling.benchmark.schemas import (
    BenchmarkArtifactBundle,
    BenchmarkConfig,
    BenchmarkResultBundle,
    BenchmarkResultRow,
    BenchmarkSortDirection,
    BenchmarkSummary,
    BenchmarkTaskType,
)
from app.modeling.benchmark.service import benchmark_dataset
from app.modeling.pycaret.schemas import (
    ExperimentArtifactBundle,
    ExperimentConfig,
    ExperimentResultBundle,
    ExperimentSetupConfig,
    ExperimentSortDirection,
    ExperimentSummary,
    ExperimentTaskType,
    MLflowTrackingMode,
    SavedModelMetadata,
)
from app.modeling.pycaret.service import PyCaretExperimentService
from app.prediction import (
    BatchPredictionRequest,
    LoadedModel,
    ModelSourceType,
    PredictionService,
    PredictionTaskType,
    SchemaValidationMode,
)
from app.profiling.schemas import ProfilingArtifactBundle, ProfilingResultSummary
from app.profiling.service import profile_dataset
from app.storage import AppJobType, build_metadata_store
from app.validation.schemas import ValidationRuleConfig
from app.validation.service import validate_dataset


class _FakeClassifier:
    classes_ = ["no", "yes"]

    def predict(self, dataframe: pd.DataFrame):
        return pd.Series(
            ["yes" if value >= 0.5 else "no" for value in dataframe["a"]],
            index=dataframe.index,
        )

    def predict_proba(self, dataframe: pd.DataFrame):
        rows = []
        for value in dataframe["a"]:
            rows.append([0.2, 0.8] if value >= 0.5 else [0.9, 0.1])
        return rows


def test_local_first_smoke_flow_persists_metadata_across_workflows(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "train.csv"
    dataset_path.write_text(
        "a,b,target\n0.9,1,1\n0.1,2,0\n0.8,3,1\n0.2,4,0\n",
        encoding="utf-8",
    )

    loaded = load_dataset(DatasetInputSpec(source_type=IngestionSourceType.CSV, path=dataset_path))
    settings = AppSettings.model_validate({"artifacts": {"root_dir": str(tmp_path / "artifacts")}})
    metadata_store = build_metadata_store(settings)
    dataset_fingerprint = loaded.metadata.content_hash or loaded.metadata.schema_hash

    assert metadata_store is not None

    validation_summary, validation_artifacts = validate_dataset(
        loaded.dataframe,
        ValidationRuleConfig(target_column="target"),
        dataset_name="train",
        artifacts_dir=settings.validation.artifacts_dir,
        loaded_dataset=loaded,
        metadata_store=metadata_store,
    )

    class _FakeProfilingService:
        def __init__(self, artifacts_dir=None):
            self._artifacts_dir = artifacts_dir

        def profile(self, df, config, dataset_name=None):
            html_path = self._artifacts_dir / "profile.html"
            summary_path = self._artifacts_dir / "profile_summary.json"
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text("<html></html>", encoding="utf-8")
            summary_path.write_text("{}", encoding="utf-8")
            return (
                ProfilingResultSummary(
                    dataset_name=dataset_name,
                    row_count=len(df.index),
                    column_count=len(df.columns),
                    numeric_column_count=3,
                    categorical_column_count=0,
                    report_mode=config.mode,
                    sampling_applied=False,
                ),
                ProfilingArtifactBundle(
                    html_report_path=html_path,
                    summary_json_path=summary_path,
                ),
            )

    monkeypatch.setattr("app.profiling.service.YDataProfilingService", _FakeProfilingService)
    profiling_summary, profiling_artifacts = profile_dataset(
        loaded.dataframe,
        mode=ProfilingMode.MINIMAL,
        dataset_name="train",
        artifacts_dir=settings.profiling.artifacts_dir,
        loaded_dataset=loaded,
        metadata_store=metadata_store,
    )

    class _FakeBenchmarkService:
        def __init__(self, *, artifacts_dir=None, **kwargs):  # noqa: ANN003
            self._artifacts_dir = artifacts_dir

        def run(self, df, config, **kwargs):  # noqa: ANN003
            leaderboard_path = self._artifacts_dir / "leaderboard.csv"
            summary_path = self._artifacts_dir / "benchmark_summary.json"
            leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
            leaderboard_path.write_text("model,score\nDummyClassifier,0.9\n", encoding="utf-8")
            summary_path.write_text("{}", encoding="utf-8")
            row = BenchmarkResultRow(
                model_name="DummyClassifier",
                task_type=BenchmarkTaskType.CLASSIFICATION,
                primary_score=0.9,
                raw_metrics={"Accuracy": 0.9},
                training_time_seconds=0.1,
                rank=1,
            )
            summary = BenchmarkSummary(
                dataset_name=kwargs.get("dataset_name"),
                dataset_fingerprint=kwargs.get("dataset_fingerprint"),
                target_column=config.target_column,
                task_type=BenchmarkTaskType.CLASSIFICATION,
                benchmark_backend=kwargs.get("execution_backend", ExecutionBackend.LOCAL),
                workspace_mode=kwargs.get("workspace_mode", WorkspaceMode.DASHBOARD),
                ranking_metric="Accuracy",
                ranking_direction=BenchmarkSortDirection.DESCENDING,
                source_row_count=len(df.index),
                source_column_count=len(df.columns),
                benchmark_row_count=len(df.index),
                feature_column_count=len(df.columns) - 1,
                train_row_count=3,
                test_row_count=1,
                model_count=1,
                best_model_name="DummyClassifier",
                best_score=0.9,
                fastest_model_name="DummyClassifier",
                fastest_model_time_seconds=0.1,
                benchmark_duration_seconds=0.2,
                split_config=config.split,
            )
            return BenchmarkResultBundle(
                dataset_name=kwargs.get("dataset_name"),
                dataset_fingerprint=kwargs.get("dataset_fingerprint"),
                config=config,
                task_type=BenchmarkTaskType.CLASSIFICATION,
                benchmark_backend=kwargs.get("execution_backend", ExecutionBackend.LOCAL),
                workspace_mode=kwargs.get("workspace_mode", WorkspaceMode.DASHBOARD),
                raw_results=pd.DataFrame({"Accuracy": [0.9]}, index=["DummyClassifier"]),
                leaderboard=[row],
                top_models=[row],
                summary=summary,
                artifacts=BenchmarkArtifactBundle(
                    leaderboard_csv_path=leaderboard_path,
                    summary_json_path=summary_path,
                ),
            )

    monkeypatch.setattr("app.modeling.benchmark.service.LazyPredictBenchmarkService", _FakeBenchmarkService)
    benchmark_bundle = benchmark_dataset(
        loaded.dataframe,
        BenchmarkConfig(target_column="target", task_type=BenchmarkTaskType.CLASSIFICATION),
        dataset_name="train",
        dataset_fingerprint=dataset_fingerprint,
        loaded_dataset=loaded,
        metadata_store=metadata_store,
        execution_backend=ExecutionBackend.LOCAL,
        workspace_mode=WorkspaceMode.DASHBOARD,
        artifacts_dir=settings.benchmark.artifacts_dir,
    )

    experiment_service = PyCaretExperimentService(
        artifacts_dir=settings.pycaret.artifacts_dir,
        models_dir=settings.pycaret.models_dir,
        snapshots_dir=settings.pycaret.snapshots_dir,
        classification_compare_metric="Accuracy",
        regression_compare_metric="R2",
        classification_tune_metric="AUC",
        regression_tune_metric="R2",
        mlflow_experiment_name=None,
        metadata_store=metadata_store,
    )
    saved_model_path = settings.pycaret.models_dir / "smoke_model.pkl"
    saved_model_metadata_path = settings.pycaret.artifacts_dir / "smoke_model_metadata.json"
    experiment_summary_path = settings.pycaret.artifacts_dir / "experiment_summary.json"
    saved_model_path.parent.mkdir(parents=True, exist_ok=True)
    saved_model_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    saved_model_path.write_bytes(b"model")
    saved_model_metadata_path.write_text("{}", encoding="utf-8")
    experiment_summary_path.write_text("{}", encoding="utf-8")

    saved_model_metadata = SavedModelMetadata(
        task_type=ExperimentTaskType.CLASSIFICATION,
        target_column="target",
        model_id="lr",
        model_name="SmokeModel",
        model_path=saved_model_path,
        dataset_fingerprint=dataset_fingerprint,
        feature_columns=["a", "b"],
        feature_dtypes={"a": "float64", "b": "int64"},
        target_dtype="int64",
    )
    experiment_summary = ExperimentSummary(
        dataset_name="train",
        dataset_fingerprint=dataset_fingerprint,
        target_column="target",
        task_type=ExperimentTaskType.CLASSIFICATION,
        execution_backend=ExecutionBackend.LOCAL,
        workspace_mode=WorkspaceMode.DASHBOARD,
        source_row_count=len(loaded.dataframe.index),
        source_column_count=len(loaded.dataframe.columns),
        feature_column_count=2,
        compare_optimize_metric="Accuracy",
        compare_ranking_direction=ExperimentSortDirection.DESCENDING,
        selected_model_name="SmokeModel",
        saved_model_name="smoke_model",
        setup_config=ExperimentSetupConfig(session_id=42).to_summary_model(
            actual_setup_kwargs={"target": "target"}
        ),
    )
    tracked_bundle = ExperimentResultBundle(
        dataset_name="train",
        dataset_fingerprint=dataset_fingerprint,
        config=ExperimentConfig(
            target_column="target",
            task_type=ExperimentTaskType.CLASSIFICATION,
            mlflow_tracking_mode=MLflowTrackingMode.OFF,
        ),
        task_type=ExperimentTaskType.CLASSIFICATION,
        execution_backend=ExecutionBackend.LOCAL,
        workspace_mode=WorkspaceMode.DASHBOARD,
        feature_columns=["a", "b"],
        feature_dtypes={"a": "float64", "b": "int64"},
        target_dtype="int64",
        saved_model_metadata=saved_model_metadata,
        artifacts=ExperimentArtifactBundle(
            summary_json_path=experiment_summary_path,
            saved_model_metadata_path=saved_model_metadata_path,
        ),
        summary=experiment_summary,
    )
    experiment_service._track_bundle(tracked_bundle)

    prediction_service = PredictionService(
        artifacts_dir=settings.prediction.artifacts_dir,
        history_path=settings.prediction.history_path,
        schema_validation_mode=SchemaValidationMode.STRICT,
        prediction_column_name=settings.prediction.prediction_column_name,
        prediction_score_column_name=settings.prediction.prediction_score_column_name,
        local_model_dirs=[settings.pycaret.models_dir],
        local_metadata_dirs=[settings.pycaret.artifacts_dir, settings.pycaret.models_dir],
        metadata_store=metadata_store,
    )
    loaded_model = LoadedModel(
        source_type=ModelSourceType.LOCAL_SAVED_MODEL,
        task_type=PredictionTaskType.CLASSIFICATION,
        model_identifier="SmokeModel",
        load_reference=str(saved_model_path),
        loader_name="TestLoader",
        scorer_kind="sklearn_like",
        supported_prediction_modes=[],
        feature_columns=["a", "b"],
        target_column="target",
        metadata={"feature_dtypes": {"a": "float64", "b": "int64"}},
        native_model=_FakeClassifier(),
    )
    monkeypatch.setattr(prediction_service, "load_model", lambda request: loaded_model)
    prediction_result = prediction_service.predict_batch(
        BatchPredictionRequest(
            source_type=ModelSourceType.LOCAL_SAVED_MODEL,
            model_identifier="SmokeModel",
            dataframe=loaded.dataframe[["a", "b"]],
            dataset_name="train",
            input_source_label="train",
        )
    )

    recent_job_types = {job.job_type for job in metadata_store.list_recent_jobs(limit=10)}
    saved_models = metadata_store.list_saved_local_models(limit=5)
    history_entries = prediction_service.list_history(limit=5)

    assert validation_summary.dataset_name == "train"
    assert validation_artifacts is not None
    assert profiling_summary.dataset_name == "train"
    assert profiling_artifacts is not None
    assert benchmark_bundle.summary.best_model_name == "DummyClassifier"
    assert prediction_result.summary.rows_scored == len(loaded.dataframe.index)
    assert {
        AppJobType.VALIDATION,
        AppJobType.PROFILING,
        AppJobType.BENCHMARK,
        AppJobType.EXPERIMENT,
        AppJobType.PREDICTION,
    }.issubset(recent_job_types)
    assert any(record.model_path == saved_model_path for record in saved_models)
    assert history_entries[0].model_identifier == "SmokeModel"


def test_streamlit_config_disables_auto_sidebar_navigation():
    """Regression: .streamlit/config.toml must disable auto-discovered sidebar navigation."""
    config_path = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"
    assert config_path.exists(), f".streamlit/config.toml not found at {config_path}"
    content = config_path.read_text(encoding="utf-8")
    assert "showSidebarNavigation" in content
    assert "false" in content.lower()