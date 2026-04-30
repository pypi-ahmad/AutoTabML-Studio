"""Focused tests for the shared modeling architecture bases."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from app.config.enums import ExecutionBackend, WorkspaceMode
from app.modeling.base import BaseService, BaseTracker
from app.modeling.benchmark.artifacts import write_benchmark_artifacts
from app.modeling.benchmark.mlflow_tracking import MLflowBenchmarkTracker
from app.modeling.benchmark.ranker import rank_result_rows
from app.modeling.benchmark.schemas import (
    BenchmarkArtifactBundle,
    BenchmarkConfig,
    BenchmarkResultBundle,
    BenchmarkSortDirection,
    BenchmarkTaskType,
)
from app.modeling.benchmark.summary import build_benchmark_summary, build_result_rows
from app.modeling.flaml.mlflow_tracking import MLflowFlamlTracker
from app.modeling.flaml.schemas import (
    FlamlArtifactBundle,
    FlamlConfig,
    FlamlResultBundle,
    FlamlSummary,
    FlamlTaskType,
)
from app.modeling.pycaret.artifacts import write_experiment_artifacts
from app.modeling.pycaret.mlflow_tracking import MLflowExperimentTracker
from app.modeling.pycaret.schemas import (
    ExperimentArtifactBundle,
    ExperimentConfig,
    ExperimentResultBundle,
    ExperimentSetupConfig,
    ExperimentSortDirection,
    ExperimentSummary,
    ExperimentTaskType,
    ExperimentTuneConfig,
)


class _FakeRun:
    def __init__(self, run_id: str = "run-123"):
        self.info = type("Info", (), {"run_id": run_id})()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeMLflow:
    def __init__(self):
        self.experiment_name = None
        self.run_name = None
        self.run_id = None
        self.params = None
        self.metrics = None
        self.artifacts: list[str] = []
        self.tracking_uri = None
        self.registry_uri = None

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_registry_uri(self, uri):
        self.registry_uri = uri

    def set_experiment(self, name):
        self.experiment_name = name

    def start_run(self, run_name=None, run_id=None):
        self.run_name = run_name
        self.run_id = run_id
        return _FakeRun()

    def log_params(self, params):
        self.params = params

    def log_metrics(self, metrics):
        self.metrics = metrics

    def log_artifact(self, path):
        self.artifacts.append(path)


class _DummyTracker(BaseTracker[object]):
    def _is_mlflow_available(self) -> bool:
        return False

    def _get_mlflow_module(self):
        raise AssertionError("not reached")

    def _operation_name(self) -> str:
        return "tests.dummy_tracker"

    def _build_run_name(self, bundle: object) -> str:
        return "dummy"

    def _build_params(self, bundle: object) -> dict[str, str]:
        return {}

    def _build_metrics(self, bundle: object) -> dict[str, float]:
        return {}

    def _artifact_paths(self, bundle: object):
        return []


class _DummyService(BaseService):
    pass


class TestBaseService:
    def test_build_tracker_and_append_warnings(self):
        service = _DummyService(
            mlflow_experiment_name="exp-name",
            tracking_uri="sqlite:///tracking.db",
            registry_uri="sqlite:///registry.db",
        )

        tracker = service._build_tracker(_DummyTracker)
        bundle = SimpleNamespace(warnings=[], summary=SimpleNamespace(warnings=[]))
        service._append_bundle_warnings(bundle, ["one", "one", "two"])

        assert tracker is not None
        assert tracker._experiment_name == "exp-name"
        assert tracker._tracking_uri == "sqlite:///tracking.db"
        assert tracker._registry_uri == "sqlite:///registry.db"
        assert bundle.warnings == ["one", "two"]
        assert bundle.summary.warnings == ["one", "two"]


class TestArchitectureArtifacts:
    def test_benchmark_artifacts_use_shared_writer_helpers(self, tmp_path: Path):
        raw_df = pd.DataFrame({"Balanced Accuracy": [0.82], "Time Taken": [0.11]}, index=["ModelA"])
        rows = build_result_rows(
            raw_df,
            task_type=BenchmarkTaskType.CLASSIFICATION,
            benchmark_backend=ExecutionBackend.LOCAL,
        )
        ranked = rank_result_rows(
            rows,
            ranking_metric="Balanced Accuracy",
            direction=BenchmarkSortDirection.DESCENDING,
        )
        summary = build_benchmark_summary(
            dataset_name="unsafe/folder:name",
            dataset_fingerprint="abc123",
            config=BenchmarkConfig(target_column="target"),
            task_type=BenchmarkTaskType.CLASSIFICATION,
            benchmark_backend=ExecutionBackend.LOCAL,
            workspace_mode=WorkspaceMode.DASHBOARD,
            ranking_metric="Balanced Accuracy",
            ranking_direction=BenchmarkSortDirection.DESCENDING,
            ranked_rows=ranked,
            source_row_count=10,
            source_column_count=4,
            benchmark_row_count=10,
            feature_column_count=3,
            train_row_count=8,
            test_row_count=2,
            sampled_row_count=None,
            stratified_split_applied=True,
            benchmark_duration_seconds=1.5,
            warnings=[],
        )
        bundle = BenchmarkResultBundle(
            dataset_name="unsafe/folder:name",
            dataset_fingerprint="abc123",
            config=BenchmarkConfig(target_column="target"),
            task_type=BenchmarkTaskType.CLASSIFICATION,
            benchmark_backend=ExecutionBackend.LOCAL,
            workspace_mode=WorkspaceMode.DASHBOARD,
            raw_results=raw_df,
            leaderboard=ranked,
            top_models=ranked,
            summary=summary,
        )

        artifacts = write_benchmark_artifacts(bundle, tmp_path)

        assert isinstance(artifacts, BenchmarkArtifactBundle)
        assert artifacts.raw_results_csv_path is not None
        assert artifacts.summary_json_path is not None
        assert "/" not in artifacts.raw_results_csv_path.name
        assert ":" not in artifacts.raw_results_csv_path.name

    def test_experiment_artifacts_use_shared_writer_helpers(self, tmp_path: Path):
        summary = ExperimentSummary(
            dataset_name="unsafe/folder:name",
            dataset_fingerprint="abc123",
            target_column="target",
            task_type=ExperimentTaskType.CLASSIFICATION,
            execution_backend=ExecutionBackend.LOCAL,
            workspace_mode=WorkspaceMode.DASHBOARD,
            source_row_count=10,
            source_column_count=4,
            feature_column_count=3,
            experiment_duration_seconds=1.5,
            setup_config=ExperimentSetupConfig(session_id=42).to_summary_model(
                actual_setup_kwargs={"target": "target", "session_id": 42}
            ),
        )
        bundle = ExperimentResultBundle(
            dataset_name="unsafe/folder:name",
            dataset_fingerprint="abc123",
            config=ExperimentConfig(target_column="target"),
            task_type=ExperimentTaskType.CLASSIFICATION,
            execution_backend=ExecutionBackend.LOCAL,
            workspace_mode=WorkspaceMode.DASHBOARD,
            summary=summary,
        )

        artifacts = write_experiment_artifacts(bundle, tmp_path)
        artifacts_second = write_experiment_artifacts(bundle, tmp_path)

        assert isinstance(artifacts, ExperimentArtifactBundle)
        assert artifacts.summary_json_path is not None
        assert artifacts.summary_json_path.exists()
        assert "/" not in artifacts.summary_json_path.name
        assert ":" not in artifacts.summary_json_path.name
        assert artifacts.summary_json_path == artifacts_second.summary_json_path


class TestArchitectureTrackers:
    def test_benchmark_tracker_uses_shared_tracker_base(self, tmp_path: Path, monkeypatch):
        raw_artifact = tmp_path / "raw.csv"
        raw_artifact.write_text("model,score\nModelA,0.9\n", encoding="utf-8")
        summary_artifact = tmp_path / "summary.json"
        summary_artifact.write_text("{}", encoding="utf-8")

        raw_df = pd.DataFrame({"Balanced Accuracy": [0.82], "Time Taken": [0.11]}, index=["ModelA"])
        rows = build_result_rows(
            raw_df,
            task_type=BenchmarkTaskType.CLASSIFICATION,
            benchmark_backend=ExecutionBackend.LOCAL,
        )
        ranked = rank_result_rows(
            rows,
            ranking_metric="Balanced Accuracy",
            direction=BenchmarkSortDirection.DESCENDING,
        )
        summary = build_benchmark_summary(
            dataset_name="housing",
            dataset_fingerprint="fp-1",
            config=BenchmarkConfig(target_column="target"),
            task_type=BenchmarkTaskType.CLASSIFICATION,
            benchmark_backend=ExecutionBackend.LOCAL,
            workspace_mode=WorkspaceMode.DASHBOARD,
            ranking_metric="Balanced Accuracy",
            ranking_direction=BenchmarkSortDirection.DESCENDING,
            ranked_rows=ranked,
            source_row_count=10,
            source_column_count=4,
            benchmark_row_count=10,
            feature_column_count=3,
            train_row_count=8,
            test_row_count=2,
            sampled_row_count=None,
            stratified_split_applied=True,
            benchmark_duration_seconds=1.5,
            warnings=[],
        )
        bundle = BenchmarkResultBundle(
            dataset_name="housing",
            dataset_fingerprint="fp-1",
            config=BenchmarkConfig(target_column="target"),
            task_type=BenchmarkTaskType.CLASSIFICATION,
            benchmark_backend=ExecutionBackend.LOCAL,
            workspace_mode=WorkspaceMode.DASHBOARD,
            raw_results=raw_df,
            leaderboard=ranked,
            top_models=ranked,
            summary=summary,
        )
        bundle.artifacts = SimpleNamespace(
            raw_results_csv_path=raw_artifact,
            leaderboard_csv_path=None,
            leaderboard_json_path=None,
            summary_json_path=summary_artifact,
            markdown_summary_path=None,
            score_chart_path=None,
            training_time_chart_path=None,
        )

        fake_mlflow = _FakeMLflow()
        monkeypatch.setattr(
            "app.modeling.benchmark.mlflow_tracking.is_mlflow_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "app.modeling.benchmark.mlflow_tracking._get_mlflow_module",
            lambda: fake_mlflow,
        )

        tracker = MLflowBenchmarkTracker(
            "autotabml-benchmarks",
            tracking_uri="sqlite:///artifacts/mlflow/mlflow.db",
            registry_uri="sqlite:///artifacts/mlflow/mlflow.db",
        )
        run_id, warnings = tracker.log_benchmark_run(bundle)

        assert warnings == []
        assert run_id == "run-123"
        assert fake_mlflow.run_name == "benchmark-classification-housing"
        assert fake_mlflow.params["target_column"] == "target"
        assert fake_mlflow.metrics["best_score"] == 0.82
        assert str(raw_artifact) in fake_mlflow.artifacts
        assert str(summary_artifact) in fake_mlflow.artifacts

    def test_experiment_tracker_uses_shared_tracker_base(self, tmp_path: Path, monkeypatch):
        summary_artifact = tmp_path / "summary.json"
        summary_artifact.write_text("{}", encoding="utf-8")

        summary = ExperimentSummary(
            dataset_name="housing",
            dataset_fingerprint="fp-1",
            target_column="target",
            task_type=ExperimentTaskType.CLASSIFICATION,
            execution_backend=ExecutionBackend.LOCAL,
            workspace_mode=WorkspaceMode.DASHBOARD,
            source_row_count=10,
            source_column_count=4,
            feature_column_count=3,
            compare_optimize_metric="Accuracy",
            compare_ranking_direction=ExperimentSortDirection.DESCENDING,
            best_baseline_model_name="Logistic Regression",
            best_baseline_score=0.81,
            tuned_model_name="Tuned Logistic Regression",
            tuned_score=0.88,
            experiment_duration_seconds=1.5,
            setup_config=ExperimentSetupConfig(session_id=42).to_summary_model(
                actual_setup_kwargs={"target": "target", "session_id": 42}
            ),
        )
        bundle = ExperimentResultBundle(
            dataset_name="housing",
            dataset_fingerprint="fp-1",
            config=ExperimentConfig(
                target_column="target",
                task_type=ExperimentTaskType.CLASSIFICATION,
                tune=ExperimentTuneConfig(optimize="AUC", n_iter=5),
            ),
            task_type=ExperimentTaskType.CLASSIFICATION,
            execution_backend=ExecutionBackend.LOCAL,
            workspace_mode=WorkspaceMode.DASHBOARD,
            summary=summary,
            artifacts=ExperimentArtifactBundle(summary_json_path=summary_artifact),
        )

        fake_mlflow = _FakeMLflow()
        monkeypatch.setattr("app.modeling.pycaret.mlflow_tracking.is_mlflow_available", lambda: True)
        monkeypatch.setattr(
            "app.modeling.pycaret.mlflow_tracking._get_mlflow_module",
            lambda: fake_mlflow,
        )

        tracker = MLflowExperimentTracker(
            "autotabml-experiments",
            tracking_uri="sqlite:///artifacts/mlflow/mlflow.db",
            registry_uri="sqlite:///artifacts/mlflow/mlflow.db",
        )
        run_id, warnings = tracker.log_experiment_bundle(bundle)

        assert warnings == []
        assert run_id == "run-123"
        assert fake_mlflow.params["task_type"] == "classification"
        assert fake_mlflow.params["tune_optimize_metric"] == "AUC"
        assert fake_mlflow.metrics["best_baseline_score"] == 0.81
        assert fake_mlflow.metrics["tuned_score"] == 0.88
        assert str(summary_artifact) in fake_mlflow.artifacts

    def test_flaml_tracker_uses_shared_tracker_base(self, tmp_path: Path, monkeypatch):
        summary_artifact = tmp_path / "summary.json"
        summary_artifact.write_text("{}", encoding="utf-8")

        bundle = FlamlResultBundle(
            dataset_name="housing",
            config=FlamlConfig(target_column="target"),
            task_type=FlamlTaskType.CLASSIFICATION,
            summary=FlamlSummary(
                dataset_name="housing",
                dataset_fingerprint="fp-1",
                target_column="target",
                task_type=FlamlTaskType.CLASSIFICATION,
                execution_backend=ExecutionBackend.LOCAL,
                workspace_mode=WorkspaceMode.DASHBOARD,
                source_row_count=10,
                source_column_count=4,
                feature_column_count=3,
                best_estimator="lgbm",
                metric="accuracy",
                search_duration_seconds=2.4,
            ),
            artifacts=FlamlArtifactBundle(summary_json_path=summary_artifact),
        )

        fake_mlflow = _FakeMLflow()
        monkeypatch.setattr("app.modeling.flaml.mlflow_tracking.is_mlflow_available", lambda: True)
        monkeypatch.setattr(
            "app.modeling.flaml.mlflow_tracking._get_mlflow_module",
            lambda: fake_mlflow,
        )

        tracker = MLflowFlamlTracker(
            "autotabml-flaml",
            tracking_uri="sqlite:///artifacts/mlflow/mlflow.db",
            registry_uri="sqlite:///artifacts/mlflow/mlflow.db",
        )
        run_id, warnings = tracker.log_flaml_bundle(bundle)

        assert warnings == []
        assert run_id == "run-123"
        assert fake_mlflow.params["framework"] == "flaml"
        assert fake_mlflow.params["task_type"] == "classification"
        assert fake_mlflow.metrics["search_duration_seconds"] == 2.4
        assert str(summary_artifact) in fake_mlflow.artifacts
