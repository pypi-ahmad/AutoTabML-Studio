from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from app.config.enums import ExecutionBackend
from app.modeling.benchmark.errors import BenchmarkExecutionError
from app.modeling.benchmark.lazypredict_runner import LazyPredictBenchmarkService
from app.modeling.benchmark.mlflow_tracking import MLflowBenchmarkTracker
from app.modeling.benchmark.schemas import BenchmarkConfig, BenchmarkTaskType
from app.modeling.flaml.mlflow_tracking import MLflowFlamlTracker
from app.modeling.flaml.schemas import FlamlTaskType
from app.modeling.pycaret.evaluate_runner import generate_evaluation_plots
from app.modeling.pycaret.mlflow_tracking import MLflowExperimentTracker
from app.modeling.pycaret.schemas import ExperimentEvaluationConfig, ExperimentTaskType, MLflowTrackingMode


class _FailingRunContext:
    def __enter__(self):
        raise RuntimeError("mlflow boom")

    def __exit__(self, exc_type, exc, tb):
        return False


def _fake_mlflow_module() -> SimpleNamespace:
    return SimpleNamespace(
        exceptions=SimpleNamespace(MlflowException=RuntimeError),
        set_tracking_uri=lambda *_args, **_kwargs: None,
        set_registry_uri=lambda *_args, **_kwargs: None,
        set_experiment=lambda *_args, **_kwargs: None,
        start_run=lambda **_kwargs: _FailingRunContext(),
        log_params=lambda *_args, **_kwargs: None,
        log_metrics=lambda *_args, **_kwargs: None,
        log_artifact=lambda *_args, **_kwargs: None,
    )


def test_benchmark_tracker_returns_warning_on_mlflow_failure(monkeypatch):
    import app.modeling.benchmark.mlflow_tracking as tracking_module

    monkeypatch.setattr(tracking_module, "is_mlflow_available", lambda: True)
    monkeypatch.setattr(tracking_module, "_get_mlflow_module", _fake_mlflow_module)

    tracker = MLflowBenchmarkTracker("benchmark-exp")
    bundle = SimpleNamespace(
        dataset_name="dataset",
        dataset_fingerprint="fingerprint",
        task_type=BenchmarkTaskType.CLASSIFICATION,
        config=BenchmarkConfig(target_column="target"),
        summary=SimpleNamespace(
            stratified_split_applied=False,
            ranking_metric="Accuracy",
            source_row_count=10,
            source_column_count=3,
            benchmark_row_count=10,
            feature_column_count=2,
            sampled_row_count=0,
            model_count=2,
            benchmark_duration_seconds=1.5,
            best_score=None,
            fastest_model_time_seconds=None,
        ),
        benchmark_backend=ExecutionBackend.LOCAL,
        workspace_mode=None,
        artifacts=None,
    )

    run_id, warnings = tracker.log_benchmark_run(bundle)

    assert run_id is None
    assert any("mlflow boom" in warning for warning in warnings)


def test_flaml_tracker_preserves_existing_run_id_on_mlflow_failure(monkeypatch):
    import app.modeling.flaml.mlflow_tracking as tracking_module

    monkeypatch.setattr(tracking_module, "is_mlflow_available", lambda: True)
    monkeypatch.setattr(tracking_module, "_get_mlflow_module", _fake_mlflow_module)

    tracker = MLflowFlamlTracker("flaml-exp")
    bundle = SimpleNamespace(
        dataset_name="dataset",
        task_type=FlamlTaskType.CLASSIFICATION,
        config=SimpleNamespace(target_column="target", search=SimpleNamespace(time_budget=30, n_splits=3)),
        execution_backend=ExecutionBackend.LOCAL,
        search_result=SimpleNamespace(best_estimator="lgbm", metric="accuracy", best_loss=None, best_config_train_time=None, time_to_find_best=None),
        summary=SimpleNamespace(search_duration_seconds=1.2),
        artifacts=None,
    )

    run_id, warnings = tracker.log_flaml_bundle(bundle, existing_run_id="existing-run")

    assert run_id == "existing-run"
    assert any("mlflow boom" in warning for warning in warnings)


def test_pycaret_tracker_preserves_existing_run_id_on_mlflow_failure(monkeypatch):
    import app.modeling.pycaret.mlflow_tracking as tracking_module

    monkeypatch.setattr(tracking_module, "is_mlflow_available", lambda: True)
    monkeypatch.setattr(tracking_module, "_get_mlflow_module", _fake_mlflow_module)

    tracker = MLflowExperimentTracker("pycaret-exp")
    bundle = SimpleNamespace(
        dataset_name="dataset",
        dataset_fingerprint="fingerprint",
        task_type=ExperimentTaskType.CLASSIFICATION,
        execution_backend=ExecutionBackend.LOCAL,
        workspace_mode=None,
        config=SimpleNamespace(target_column="target", mlflow_tracking_mode=MLflowTrackingMode.MANUAL),
        summary=SimpleNamespace(
            experiment_duration_seconds=3.2,
            compare_optimize_metric="Accuracy",
            tune_optimize_metric="Accuracy",
            selected_model_id="lr",
            selected_model_name="Logistic Regression",
            best_baseline_model_name="Logistic Regression",
            tuned_model_name="Logistic Regression",
            best_baseline_score=None,
            tuned_score=None,
            setup_config=SimpleNamespace(actual_setup_kwargs={}),
        ),
        compare_leaderboard=[],
        artifacts=None,
    )

    run_id, warnings = tracker.log_experiment_bundle(bundle, existing_run_id="existing-run")

    assert run_id == "existing-run"
    assert any("mlflow boom" in warning for warning in warnings)


def test_lazypredict_service_wraps_split_failure(monkeypatch):
    import app.modeling.benchmark.lazypredict_runner as runner_module

    monkeypatch.setattr(runner_module, "is_lazypredict_available", lambda: True)

    def _raise_split_error(*_args, **_kwargs):
        raise ValueError("bad split")

    monkeypatch.setattr(runner_module, "_train_test_split", _raise_split_error)

    service = LazyPredictBenchmarkService()
    df = pd.DataFrame(
        {
            "feature_num": [1, 2, 3, 4, 5, 6],
            "feature_cat": ["a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    with pytest.raises(BenchmarkExecutionError, match="Failed to create the train/test split: bad split"):
        service.run(
            df,
            BenchmarkConfig(target_column="target", task_type=BenchmarkTaskType.CLASSIFICATION),
        )


def test_generate_evaluation_plots_returns_warnings_for_runtime_failures(tmp_path):
    class FakeExperimentHandle:
        def plot_model(self, *_args, **_kwargs):
            raise RuntimeError("plot failure")

        def evaluate_model(self, *_args, **_kwargs):
            raise RuntimeError("interactive failure")

    artifacts, warnings = generate_evaluation_plots(
        FakeExperimentHandle(),
        object(),
        task_type=ExperimentTaskType.CLASSIFICATION,
        model_name="best-model",
        evaluation=ExperimentEvaluationConfig(plots=["confusion_matrix"], interactive=True),
        output_dir=tmp_path,
    )

    assert artifacts == []
    assert any("plot failure" in warning for warning in warnings)
    assert any("interactive failure" in warning for warning in warnings)