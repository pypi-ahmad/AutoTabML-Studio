"""Tests for the benchmarking foundation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.config.enums import ExecutionBackend, WorkspaceMode
from app.config.models import BenchmarkSettings
from app.modeling.benchmark.artifacts import write_benchmark_artifacts
from app.modeling.benchmark.errors import BenchmarkTargetError
from app.modeling.benchmark.lazypredict_runner import LazyPredictBenchmarkService
from app.modeling.benchmark.mlflow_tracking import MLflowBenchmarkTracker
from app.modeling.benchmark.ranker import rank_result_rows, resolve_ranking_metric
from app.modeling.benchmark.schemas import (
    BenchmarkConfig,
    BenchmarkResultBundle,
    BenchmarkResultRow,
    BenchmarkSortDirection,
    BenchmarkSplitConfig,
    BenchmarkTaskType,
)
from app.modeling.benchmark.summary import (
    build_benchmark_summary,
    build_result_rows,
)
from app.pages.benchmark_page import (
    build_benchmark_run_key,
    default_ranking_metric_for_task,
)


@pytest.fixture
def classification_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_num": [1, 2, 3, 4, 5, 6],
            "feature_cat": ["a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def regression_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_num": [1, 2, 3, 4, 5, 6],
            "feature_cat": ["a", "b", "c", "d", "e", "f"],
            "target": [10.1, 20.2, 30.3, 40.4, 50.5, 60.6],
        }
    )


class _FakeEstimator:
    pass


class _FakeLazyClassifier:
    init_kwargs = None
    fit_args = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def fit(self, X_train, X_test, y_train, y_test):
        type(self).fit_args = (X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
        return pd.DataFrame(
            {
                "Accuracy": [0.80, 0.75],
                "Balanced Accuracy": [0.82, 0.70],
                "ROC AUC": [0.84, 0.71],
                "F1 Score": [0.81, 0.69],
                "Precision": [0.80, 0.70],
                "Recall": [0.82, 0.68],
                "Time Taken": [0.10, 0.05],
            },
            index=["ModelA", "ModelB"],
        )


class _FakeLazyRegressor:
    init_kwargs = None
    fit_args = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def fit(self, X_train, X_test, y_train, y_test):
        type(self).fit_args = (X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy())
        return pd.DataFrame(
            {
                "Adjusted R-Squared": [0.92, 0.85],
                "R-Squared": [0.93, 0.86],
                "RMSE": [1.1, 2.3],
                "Time Taken": [0.21, 0.09],
            },
            index=["ModelR1", "ModelR2"],
        )


class _FakeLazyModule:
    LazyClassifier = _FakeLazyClassifier
    LazyRegressor = _FakeLazyRegressor
    CLASSIFIERS = [("ModelA", _FakeEstimator), ("ModelB", _FakeEstimator)]
    REGRESSORS = [("ModelR1", _FakeEstimator), ("ModelR2", _FakeEstimator)]


def _fake_train_test_split(X, y, *, test_size, random_state, stratify):
    _fake_train_test_split.kwargs = {
        "test_size": test_size,
        "random_state": random_state,
        "stratify": stratify,
    }
    split_index = max(1, int(round(len(X) * (1 - test_size))))
    return X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]


def _reset_fake_lazy_state() -> None:
    _FakeLazyClassifier.init_kwargs = None
    _FakeLazyClassifier.fit_args = None
    _FakeLazyRegressor.init_kwargs = None
    _FakeLazyRegressor.fit_args = None
    _fake_train_test_split.kwargs = {}


class TestBenchmarkRouting:
    def test_routes_classification_to_lazy_classifier(self, classification_df, monkeypatch):
        _reset_fake_lazy_state()
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner.is_lazypredict_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner._lazypredict_gpu_usable",
            lambda: True,
        )
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner._get_lazypredict_module",
            lambda: _FakeLazyModule,
        )
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner._train_test_split",
            _fake_train_test_split,
        )

        service = LazyPredictBenchmarkService(artifacts_dir=None, mlflow_experiment_name=None)
        bundle = service.run(
            classification_df,
            BenchmarkConfig(
                target_column="target",
                task_type=BenchmarkTaskType.CLASSIFICATION,
                split=BenchmarkSplitConfig(test_size=0.33, random_state=7),
            ),
            dataset_name="cls",
            execution_backend=ExecutionBackend.LOCAL,
            workspace_mode=WorkspaceMode.DASHBOARD,
        )

        assert _FakeLazyClassifier.init_kwargs is not None
        assert _FakeLazyRegressor.init_kwargs is None
        assert _FakeLazyClassifier.init_kwargs["use_gpu"] is True
        assert bundle.task_type == BenchmarkTaskType.CLASSIFICATION
        assert bundle.summary.best_model_name == "ModelA"
        assert any("GPU acceleration is enabled" in warning for warning in bundle.warnings)

    def test_routes_regression_to_lazy_regressor(self, regression_df, monkeypatch):
        _reset_fake_lazy_state()
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner.is_lazypredict_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner._lazypredict_gpu_usable",
            lambda: False,
        )
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner._get_lazypredict_module",
            lambda: _FakeLazyModule,
        )
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner._train_test_split",
            _fake_train_test_split,
        )

        service = LazyPredictBenchmarkService(artifacts_dir=None, mlflow_experiment_name=None)
        bundle = service.run(
            regression_df,
            BenchmarkConfig(
                target_column="target",
                task_type=BenchmarkTaskType.REGRESSION,
            ),
            dataset_name="reg",
            execution_backend=ExecutionBackend.LOCAL,
        )

        assert _FakeLazyRegressor.init_kwargs is not None
        assert _FakeLazyClassifier.init_kwargs is None
        assert _FakeLazyRegressor.init_kwargs["use_gpu"] is False
        assert bundle.task_type == BenchmarkTaskType.REGRESSION
        assert bundle.summary.ranking_metric == "Adjusted R-Squared"
        assert any("benchmarking will run on CPU" in warning for warning in bundle.warnings)

    def test_respects_explicit_cpu_preference_even_when_cuda_available(self, classification_df, monkeypatch):
        _reset_fake_lazy_state()
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner.is_lazypredict_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner._lazypredict_gpu_usable",
            lambda: True,
        )
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner._get_lazypredict_module",
            lambda: _FakeLazyModule,
        )
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner._train_test_split",
            _fake_train_test_split,
        )

        service = LazyPredictBenchmarkService(artifacts_dir=None, mlflow_experiment_name=None)
        service.run(
            classification_df,
            BenchmarkConfig(
                target_column="target",
                task_type=BenchmarkTaskType.CLASSIFICATION,
                prefer_gpu=False,
            ),
            dataset_name="cls",
            execution_backend=ExecutionBackend.LOCAL,
        )

        assert _FakeLazyClassifier.init_kwargs is not None
        assert _FakeLazyClassifier.init_kwargs["use_gpu"] is False


class TestTargetValidation:
    def test_invalid_classification_target_raises(self, monkeypatch):
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner.is_lazypredict_available",
            lambda: True,
        )
        service = LazyPredictBenchmarkService(artifacts_dir=None, mlflow_experiment_name=None)
        df = pd.DataFrame({"feature": [1, 2, 3], "target": [1, 1, 1]})

        with pytest.raises(BenchmarkTargetError, match="at least two classes"):
            service.run(
                df,
                BenchmarkConfig(target_column="target", task_type=BenchmarkTaskType.CLASSIFICATION),
            )


class TestSplitBehavior:
    def test_classification_split_uses_stratify_when_reasonable(self, classification_df, monkeypatch):
        _reset_fake_lazy_state()
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner.is_lazypredict_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner._get_lazypredict_module",
            lambda: _FakeLazyModule,
        )
        monkeypatch.setattr(
            "app.modeling.benchmark.lazypredict_runner._train_test_split",
            _fake_train_test_split,
        )

        service = LazyPredictBenchmarkService(artifacts_dir=None, mlflow_experiment_name=None)
        service.run(
            classification_df,
            BenchmarkConfig(
                target_column="target",
                task_type=BenchmarkTaskType.CLASSIFICATION,
                split=BenchmarkSplitConfig(test_size=0.34, random_state=42),
            ),
        )

        assert _fake_train_test_split.kwargs["stratify"] is not None


class TestNormalization:
    def test_build_result_rows_normalizes_lazy_output(self):
        raw_df = pd.DataFrame(
            {
                "Accuracy": [0.8],
                "Balanced Accuracy": [0.82],
                "Time Taken": [0.11],
            },
            index=["ModelA"],
        )

        rows = build_result_rows(
            raw_df,
            task_type=BenchmarkTaskType.CLASSIFICATION,
            benchmark_backend=ExecutionBackend.LOCAL,
        )

        assert len(rows) == 1
        assert rows[0].model_name == "ModelA"
        assert rows[0].training_time_seconds == 0.11
        assert rows[0].raw_metrics["Balanced Accuracy"] == 0.82


class TestRanking:
    def test_rank_result_rows_descending(self):
        rows = [
            BenchmarkResultRow(
                model_name="ModelA",
                task_type=BenchmarkTaskType.CLASSIFICATION,
                raw_metrics={"Balanced Accuracy": 0.75},
                training_time_seconds=0.3,
            ),
            BenchmarkResultRow(
                model_name="ModelB",
                task_type=BenchmarkTaskType.CLASSIFICATION,
                raw_metrics={"Balanced Accuracy": 0.91},
                training_time_seconds=0.4,
            ),
        ]

        ranked = rank_result_rows(
            rows,
            ranking_metric="Balanced Accuracy",
            direction=BenchmarkSortDirection.DESCENDING,
        )

        assert ranked[0].model_name == "ModelB"
        assert ranked[0].rank == 1
        assert ranked[0].primary_score == 0.91

    def test_rank_result_rows_ascending_rmse(self):
        rows = [
            BenchmarkResultRow(
                model_name="ModelHigh",
                task_type=BenchmarkTaskType.REGRESSION,
                raw_metrics={"RMSE": 5.0},
                training_time_seconds=0.2,
            ),
            BenchmarkResultRow(
                model_name="ModelLow",
                task_type=BenchmarkTaskType.REGRESSION,
                raw_metrics={"RMSE": 1.5},
                training_time_seconds=0.3,
            ),
        ]

        ranked = rank_result_rows(
            rows,
            ranking_metric="RMSE",
            direction=BenchmarkSortDirection.ASCENDING,
        )

        assert ranked[0].model_name == "ModelLow"
        assert ranked[0].rank == 1
        assert ranked[0].primary_score == 1.5
        assert ranked[1].model_name == "ModelHigh"

    def test_fallback_metric_logic(self):
        metric, direction, warnings = resolve_ranking_metric(
            BenchmarkTaskType.REGRESSION,
            ["RMSE", "Time Taken"],
            preferred_metric="Accuracy",
            default_metric="Adjusted R-Squared",
        )

        assert metric == "RMSE"
        assert direction == BenchmarkSortDirection.ASCENDING
        assert warnings


class TestArtifacts:
    def test_artifact_path_generation(self, tmp_path: Path):
        raw_df = pd.DataFrame(
            {"Balanced Accuracy": [0.82], "Time Taken": [0.11]},
            index=["ModelA"],
        )
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

        assert artifacts.raw_results_csv_path is not None
        assert artifacts.raw_results_csv_path.exists()
        assert "/" not in artifacts.raw_results_csv_path.name
        assert ":" not in artifacts.raw_results_csv_path.name
        assert artifacts.leaderboard_csv_path is not None
        assert artifacts.summary_json_path is not None


class TestMLflowTracking:
    def test_mlflow_wrapper_logs_params_metrics_and_artifacts(self, tmp_path: Path, monkeypatch):
        raw_artifact = tmp_path / "raw.csv"
        raw_artifact.write_text("model,score\nModelA,0.9\n", encoding="utf-8")
        summary_artifact = tmp_path / "summary.json"
        summary_artifact.write_text("{}", encoding="utf-8")

        raw_df = pd.DataFrame(
            {"Balanced Accuracy": [0.82], "Time Taken": [0.11]},
            index=["ModelA"],
        )
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
        bundle.artifacts = type("Artifacts", (), {
            "raw_results_csv_path": raw_artifact,
            "leaderboard_csv_path": None,
            "leaderboard_json_path": None,
            "summary_json_path": summary_artifact,
            "markdown_summary_path": None,
            "score_chart_path": None,
            "training_time_chart_path": None,
        })()

        class _FakeRun:
            def __init__(self):
                self.info = type("Info", (), {"run_id": "run-123"})()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _FakeMLflow:
            def __init__(self):
                self.experiment_name = None
                self.run_name = None
                self.params = None
                self.metrics = None
                self.artifacts = []
                self.tracking_uri = None
                self.registry_uri = None

            def set_tracking_uri(self, uri):
                self.tracking_uri = uri

            def set_registry_uri(self, uri):
                self.registry_uri = uri

            def set_experiment(self, name):
                self.experiment_name = name

            def start_run(self, run_name):
                self.run_name = run_name
                return _FakeRun()

            def log_params(self, params):
                self.params = params

            def log_metrics(self, metrics):
                self.metrics = metrics

            def log_artifact(self, path):
                self.artifacts.append(path)

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
        assert fake_mlflow.tracking_uri == "sqlite:///artifacts/mlflow/mlflow.db"
        assert fake_mlflow.registry_uri == "sqlite:///artifacts/mlflow/mlflow.db"
        assert fake_mlflow.experiment_name == "autotabml-benchmarks"
        assert fake_mlflow.run_name == "benchmark-classification-housing"
        assert fake_mlflow.params["target_column"] == "target"
        assert fake_mlflow.metrics["best_score"] == 0.82
        assert str(raw_artifact) in fake_mlflow.artifacts
        assert str(summary_artifact) in fake_mlflow.artifacts


class TestSummaryGeneration:
    def test_build_benchmark_summary(self):
        rows = [
            BenchmarkResultRow(
                model_name="BestModel",
                task_type=BenchmarkTaskType.REGRESSION,
                primary_score=0.95,
                raw_metrics={"Adjusted R-Squared": 0.95},
                training_time_seconds=0.42,
                rank=1,
            ),
            BenchmarkResultRow(
                model_name="FastModel",
                task_type=BenchmarkTaskType.REGRESSION,
                primary_score=0.80,
                raw_metrics={"Adjusted R-Squared": 0.80},
                training_time_seconds=0.10,
                rank=2,
            ),
        ]

        summary = build_benchmark_summary(
            dataset_name="regression_ds",
            dataset_fingerprint="fp-2",
            config=BenchmarkConfig(target_column="target", task_type=BenchmarkTaskType.REGRESSION),
            task_type=BenchmarkTaskType.REGRESSION,
            benchmark_backend=ExecutionBackend.LOCAL,
            workspace_mode=WorkspaceMode.DASHBOARD,
            ranking_metric="Adjusted R-Squared",
            ranking_direction=BenchmarkSortDirection.DESCENDING,
            ranked_rows=rows,
            source_row_count=100,
            source_column_count=8,
            benchmark_row_count=100,
            feature_column_count=7,
            train_row_count=80,
            test_row_count=20,
            sampled_row_count=None,
            stratified_split_applied=False,
            benchmark_duration_seconds=2.5,
            warnings=["example warning"],
        )

        assert summary.best_model_name == "BestModel"
        assert summary.best_score == 0.95
        assert summary.fastest_model_name == "FastModel"
        assert summary.fastest_model_time_seconds == 0.10
        assert summary.model_count == 2


class TestPageHelpers:
    def test_build_benchmark_run_key_varies_by_target_and_task(self):
        key_a = build_benchmark_run_key("housing", "price", BenchmarkTaskType.AUTO)
        key_b = build_benchmark_run_key("housing", "price", BenchmarkTaskType.REGRESSION)
        key_c = build_benchmark_run_key("housing", "target", BenchmarkTaskType.AUTO)

        assert key_a != key_b
        assert key_a != key_c

    def test_default_ranking_metric_for_task(self):
        settings = BenchmarkSettings()

        assert (
            default_ranking_metric_for_task(BenchmarkTaskType.CLASSIFICATION, settings)
            == settings.default_classification_ranking_metric
        )
        assert (
            default_ranking_metric_for_task(BenchmarkTaskType.REGRESSION, settings)
            == settings.default_regression_ranking_metric
        )
        assert default_ranking_metric_for_task(BenchmarkTaskType.AUTO, settings) == ""


class TestBenchmarkConfigDefaults:
    def test_benchmark_settings_defaults(self):
        settings = BenchmarkSettings()

        assert settings.artifacts_dir == Path("artifacts/benchmark")
        assert settings.default_test_size == 0.2
        assert settings.default_classification_ranking_metric == "Balanced Accuracy"
        assert settings.default_regression_ranking_metric == "Adjusted R-Squared"
        assert settings.mlflow_experiment_name == "autotabml-benchmarks"


class TestDegenerateAdjustedRSquared:
    """Verify that degenerate Adjusted R-Squared values are detected and skipped."""

    def test_degenerate_adjusted_r2_falls_back_to_r_squared(self):
        """When Adjusted R-Squared > 1.0, fall back to R-Squared."""
        raw_results = pd.DataFrame(
            {
                "Adjusted R-Squared": [192.4, 147.4, 100.9],
                "R-Squared": [-126.6, -96.6, -65.6],
                "RMSE": [52.9, 46.3, 38.3],
                "Time Taken": [0.01, 0.04, 0.01],
            },
            index=["KernelRidge", "MLPRegressor", "LinearSVR"],
        )

        metric, direction, warnings = resolve_ranking_metric(
            BenchmarkTaskType.REGRESSION,
            raw_results.columns,
            default_metric="Adjusted R-Squared",
            raw_results=raw_results,
        )

        assert metric != "Adjusted R-Squared"
        assert any("degenerate" in w.lower() for w in warnings)

    def test_valid_adjusted_r2_not_skipped(self):
        """Normal Adjusted R-Squared values should be used without issue."""
        raw_results = pd.DataFrame(
            {
                "Adjusted R-Squared": [0.85, 0.72, 0.60],
                "R-Squared": [0.86, 0.73, 0.61],
                "RMSE": [5.0, 7.0, 9.0],
                "Time Taken": [0.1, 0.2, 0.3],
            },
            index=["ModelA", "ModelB", "ModelC"],
        )

        metric, direction, warnings = resolve_ranking_metric(
            BenchmarkTaskType.REGRESSION,
            raw_results.columns,
            default_metric="Adjusted R-Squared",
            raw_results=raw_results,
        )

        assert metric == "Adjusted R-Squared"
        assert not any("degenerate" in w.lower() for w in warnings)