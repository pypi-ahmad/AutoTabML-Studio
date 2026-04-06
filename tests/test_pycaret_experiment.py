"""Tests for the PyCaret experiment lab foundation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.config.enums import ExecutionBackend, WorkspaceMode
from app.config.models import PyCaretExperimentSettings
from app.modeling.pycaret.artifacts import write_experiment_artifacts
from app.modeling.pycaret.errors import PyCaretConfigurationError
from app.modeling.pycaret.mlflow_tracking import MLflowExperimentTracker
from app.modeling.pycaret.persistence import build_saved_model_metadata
from app.modeling.pycaret.schemas import (
    CustomMetricSpec,
    ExperimentArtifactBundle,
    ExperimentCompareConfig,
    ExperimentConfig,
    ExperimentEvaluationConfig,
    ExperimentPersistenceConfig,
    ExperimentResultBundle,
    ExperimentSetupConfig,
    ExperimentSortDirection,
    ExperimentSummary,
    ExperimentTaskType,
    ExperimentTuneConfig,
    MLflowTrackingMode,
    ModelSelectionSpec,
)
from app.modeling.pycaret.service import PyCaretExperimentService
from app.pages.experiment_page import (
    build_experiment_run_key,
    default_compare_metric_for_task,
    default_tune_metric_for_task,
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


class _FakeExperimentBase:
    pull_queue: list[pd.DataFrame]
    added_metrics: list[tuple]
    removed_metrics: list[str]
    plotted: list[str]
    evaluated_models: list[object]
    saved_models: list[tuple]
    saved_experiments: list[Path]

    def __init__(self) -> None:
        self.pull_queue = []
        self.added_metrics = []
        self.removed_metrics = []
        self.plotted = []
        self.evaluated_models = []
        self.saved_models = []
        self.saved_experiments = []
        self.setup_kwargs = None

    def setup(self, data, target, **kwargs):
        self.setup_kwargs = {"target": target, **kwargs}
        self.data = data.copy()
        return self

    def models(self, type=None, internal=False, raise_errors=True):
        return pd.DataFrame(
            {
                "Name": ["Logistic Regression", "Random Forest Classifier", "Linear Regression"],
            },
            index=["lr", "rf", "linreg"],
        )

    def get_metrics(self, reset=False, include_custom=True, raise_errors=True):
        return pd.DataFrame(
            {
                "Name": ["Accuracy", "AUC", "R2"],
                "Display Name": ["Accuracy", "AUC", "R2"],
                "Score Function": ["accuracy_score", "roc_auc_score", "r2_score"],
                "Greater is Better": [True, True, True],
                "Custom": [False, False, False],
            },
            index=["Accuracy", "AUC", "R2"],
        )

    def add_metric(self, *args, **kwargs):
        self.added_metrics.append((args, kwargs))
        return pd.Series({"ID": args[0], "Name": args[1]})

    def remove_metric(self, name_or_id):
        self.removed_metrics.append(name_or_id)

    def pull(self, pop=False):
        if not self.pull_queue:
            return pd.DataFrame()
        return self.pull_queue.pop(0)

    def evaluate_model(self, estimator, **kwargs):
        self.evaluated_models.append(estimator)

    def plot_model(self, estimator, plot="auc", save=False, **kwargs):
        self.plotted.append(plot)
        if plot == "unsupported_plot":
            raise ValueError("plot not supported")
        output_dir = Path.cwd()
        path = output_dir / f"{plot}.png"
        path.write_bytes(b"fake-png")
        return str(path)

    def finalize_model(self, estimator, **kwargs):
        return {"finalized": estimator}

    def save_model(self, model, model_name, model_only=False, verbose=True, **kwargs):
        path = Path(f"{model_name}.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-model")
        self.saved_models.append((model, model_name, model_only))
        return model, str(path)

    def load_model(self, model_name, platform=None, authentication=None, verbose=True):
        return {"loaded_model": model_name}

    def save_experiment(self, path_or_file, **cloudpickle_kwargs):
        path = Path(path_or_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-experiment")
        self.saved_experiments.append(path)


class _FakeClassificationExperiment(_FakeExperimentBase):
    def compare_models(self, **kwargs):
        self.pull_queue.append(
            pd.DataFrame(
                {
                    "Model": ["Random Forest Classifier", "Logistic Regression"],
                    "Accuracy": [0.91, 0.88],
                    "AUC": [0.93, 0.89],
                    "Recall": [0.90, 0.86],
                    "Prec.": [0.92, 0.87],
                    "F1": [0.91, 0.86],
                    "Kappa": [0.82, 0.76],
                    "MCC": [0.83, 0.77],
                    "TT (Sec)": [0.12, 0.03],
                }
            )
        )
        return ["rf-model", "lr-model"] if kwargs.get("n_select", 1) > 1 else "rf-model"

    def create_model(self, estimator, **kwargs):
        self.pull_queue.append(
            pd.DataFrame(
                {
                    "Accuracy": [0.80, 0.82, 0.81, 0.81, 0.01],
                    "AUC": [0.84, 0.85, 0.86, 0.85, 0.01],
                    "F1": [0.79, 0.81, 0.80, 0.80, 0.01],
                },
                index=["Fold1", "Fold2", "Fold3", "Mean", "Std"],
            )
        )
        return {"created_model": estimator}

    def tune_model(self, estimator, return_tuner=False, **kwargs):
        self.pull_queue.append(
            pd.DataFrame(
                {
                    "Accuracy": [0.84, 0.85, 0.86, 0.85, 0.01],
                    "AUC": [0.87, 0.88, 0.89, 0.88, 0.01],
                    "F1": [0.83, 0.84, 0.85, 0.84, 0.01],
                },
                index=["Fold1", "Fold2", "Fold3", "Mean", "Std"],
            )
        )
        tuned = {"tuned_model": estimator}
        if return_tuner:
            return tuned, {"tuner": "fake"}
        return tuned


class _FakeRegressionExperiment(_FakeExperimentBase):
    def compare_models(self, **kwargs):
        self.pull_queue.append(
            pd.DataFrame(
                {
                    "Model": ["Linear Regression", "Random Forest Regressor"],
                    "MAE": [1.2, 1.4],
                    "MSE": [2.1, 2.8],
                    "RMSE": [1.45, 1.67],
                    "R2": [0.91, 0.88],
                    "RMSLE": [0.08, 0.10],
                    "MAPE": [0.04, 0.05],
                    "TT (Sec)": [0.02, 0.20],
                }
            )
        )
        return "linreg-model"

    def create_model(self, estimator, **kwargs):
        self.pull_queue.append(
            pd.DataFrame(
                {
                    "MAE": [1.2, 1.1, 1.15, 1.15, 0.04],
                    "RMSE": [1.5, 1.4, 1.45, 1.45, 0.05],
                    "R2": [0.88, 0.90, 0.89, 0.89, 0.01],
                },
                index=["Fold1", "Fold2", "Fold3", "Mean", "Std"],
            )
        )
        return {"created_model": estimator}

    def tune_model(self, estimator, return_tuner=False, **kwargs):
        self.pull_queue.append(
            pd.DataFrame(
                {
                    "MAE": [1.0, 1.1, 1.0, 1.03, 0.03],
                    "RMSE": [1.3, 1.4, 1.3, 1.33, 0.04],
                    "R2": [0.92, 0.91, 0.93, 0.92, 0.01],
                },
                index=["Fold1", "Fold2", "Fold3", "Mean", "Std"],
            )
        )
        tuned = {"tuned_model": estimator}
        if return_tuner:
            return tuned, {"tuner": "fake"}
        return tuned


def _make_service(monkeypatch, task_type: ExperimentTaskType):
    monkeypatch.setattr("app.modeling.pycaret.setup_runner.is_pycaret_available", lambda: True)
    monkeypatch.setattr(
        "app.modeling.pycaret.setup_runner.build_pycaret_experiment",
        lambda resolved_task_type: _FakeClassificationExperiment()
        if resolved_task_type == ExperimentTaskType.CLASSIFICATION
        else _FakeRegressionExperiment(),
    )
    return PyCaretExperimentService(
        artifacts_dir=None,
        models_dir=None,
        snapshots_dir=None,
        classification_compare_metric="Accuracy",
        regression_compare_metric="R2",
        classification_tune_metric="AUC",
        regression_tune_metric="R2",
        mlflow_experiment_name=None,
    )


class TestExperimentRouting:
    def test_routes_classification_to_classification_experiment(self, classification_df, monkeypatch):
        service = _make_service(monkeypatch, ExperimentTaskType.CLASSIFICATION)
        bundle = service.setup_experiment(
            classification_df,
            ExperimentConfig(target_column="target", task_type=ExperimentTaskType.CLASSIFICATION),
            dataset_name="cls",
            execution_backend=ExecutionBackend.LOCAL,
            workspace_mode=WorkspaceMode.DASHBOARD,
        )

        assert bundle.task_type == ExperimentTaskType.CLASSIFICATION
        assert bundle.summary.setup_config.actual_setup_kwargs["target"] == "target"

    def test_routes_regression_to_regression_experiment(self, regression_df, monkeypatch):
        service = _make_service(monkeypatch, ExperimentTaskType.REGRESSION)
        bundle = service.setup_experiment(
            regression_df,
            ExperimentConfig(target_column="target", task_type=ExperimentTaskType.REGRESSION),
            dataset_name="reg",
            execution_backend=ExecutionBackend.LOCAL,
        )

        assert bundle.task_type == ExperimentTaskType.REGRESSION
        assert bundle.summary.setup_config.actual_setup_kwargs["target"] == "target"


class TestSetupConfigNormalization:
    def test_setup_config_prefers_gpu_by_default(self):
        assert ExperimentSetupConfig().use_gpu is True

    def test_manual_mlflow_mode_keeps_pycaret_logging_off(self, classification_df, monkeypatch):
        service = _make_service(monkeypatch, ExperimentTaskType.CLASSIFICATION)
        bundle = service.setup_experiment(
            classification_df,
            ExperimentConfig(
                target_column="target",
                task_type=ExperimentTaskType.CLASSIFICATION,
                mlflow_tracking_mode=MLflowTrackingMode.MANUAL,
                setup=ExperimentSetupConfig(
                    session_id=7,
                    train_size=0.8,
                    fold=3,
                    fold_strategy="stratifiedkfold",
                    preprocess=False,
                    ignore_features=["feature_cat"],
                    log_experiment=False,
                    log_plots=["auc"],
                    log_profile=True,
                    log_data=True,
                ),
            ),
        )

        actual = bundle.summary.setup_config.actual_setup_kwargs
        assert actual["session_id"] == 7
        assert actual["train_size"] == 0.8
        assert actual["fold"] == 3
        assert actual["fold_strategy"] == "stratifiedkfold"
        assert actual["preprocess"] is False
        assert actual["ignore_features"] == ["feature_cat"]
        assert actual["log_experiment"] is False
        assert actual["log_plots"] is False
        assert actual["log_profile"] is False
        assert actual["log_data"] is False
        assert actual["html"] is False

    def test_explicit_test_data_is_captured_without_train_size(self, classification_df, monkeypatch):
        service = _make_service(monkeypatch, ExperimentTaskType.CLASSIFICATION)
        train_df = classification_df.iloc[:4].reset_index(drop=True)
        test_df = classification_df.iloc[4:].reset_index(drop=True)

        bundle = service.setup_experiment(
            train_df,
            ExperimentConfig(target_column="target", task_type=ExperimentTaskType.CLASSIFICATION),
            test_df=test_df,
        )

        actual = bundle.summary.setup_config.actual_setup_kwargs
        runtime = bundle.runtime

        assert actual["test_data_supplied"] is True
        assert actual["train_size"] is None
        assert runtime is not None
        assert "train_size" not in runtime.experiment_handle.setup_kwargs
        assert runtime.experiment_handle.setup_kwargs["test_data"].equals(test_df)
        # test_data_supplied must NOT leak into the kwargs passed to PyCaret setup()
        assert "test_data_supplied" not in runtime.experiment_handle.setup_kwargs


class TestMetricManagement:
    def test_metric_listing_and_custom_metric_registration(self, classification_df, monkeypatch):
        service = _make_service(monkeypatch, ExperimentTaskType.CLASSIFICATION)
        bundle = service.setup_experiment(
            classification_df,
            ExperimentConfig(target_column="target", task_type=ExperimentTaskType.CLASSIFICATION),
        )

        assert any(metric.metric_id == "Accuracy" for metric in bundle.available_metrics)

        spec = CustomMetricSpec(
            metric_id="logloss",
            display_name="Log Loss",
            target="pred_proba",
            greater_is_better=False,
            kwargs={"labels": [0, 1]},
        )

        service.add_custom_metric(bundle, spec, score_func=lambda y_true, y_pred, **kwargs: 0.1)
        service.remove_custom_metric(bundle, "logloss")

        runtime = bundle.runtime
        assert runtime is not None
        assert runtime.experiment_handle.added_metrics
        assert runtime.experiment_handle.removed_metrics == ["logloss"]

    def test_regression_custom_metric_rejects_classification_target(self, regression_df, monkeypatch):
        service = _make_service(monkeypatch, ExperimentTaskType.REGRESSION)
        bundle = service.setup_experiment(
            regression_df,
            ExperimentConfig(target_column="target", task_type=ExperimentTaskType.REGRESSION),
        )

        with pytest.raises(PyCaretConfigurationError, match="target='pred'"):
            service.add_custom_metric(
                bundle,
                CustomMetricSpec(
                    metric_id="bad_metric",
                    display_name="Bad Metric",
                    target="pred_proba",
                ),
                score_func=lambda y_true, y_pred, **kwargs: 0.0,
            )

    def test_reserved_custom_metric_kwargs_are_rejected(self):
        with pytest.raises(ValueError, match="reserved names"):
            CustomMetricSpec(
                metric_id="unsafe_metric",
                display_name="Unsafe Metric",
                kwargs={"score_func": "override"},
            )


class TestCompareNormalization:
    def test_compare_models_normalizes_classification_leaderboard(self, classification_df, monkeypatch):
        service = _make_service(monkeypatch, ExperimentTaskType.CLASSIFICATION)
        bundle = service.setup_experiment(
            classification_df,
            ExperimentConfig(
                target_column="target",
                task_type=ExperimentTaskType.CLASSIFICATION,
                compare=ExperimentCompareConfig(optimize="Accuracy", n_select=2),
            ),
        )

        bundle = service.compare_models(bundle)

        assert bundle.compare_leaderboard[0].model_name == "Random Forest Classifier"
        assert bundle.compare_leaderboard[0].model_id == "rf"
        assert bundle.compare_leaderboard[0].primary_score == 0.91
        assert bundle.summary.compare_optimize_metric == "Accuracy"
        assert bundle.summary.compare_ranking_direction == ExperimentSortDirection.DESCENDING

    def test_compare_models_normalizes_regression_leaderboard(self, regression_df, monkeypatch):
        service = _make_service(monkeypatch, ExperimentTaskType.REGRESSION)
        bundle = service.setup_experiment(
            regression_df,
            ExperimentConfig(
                target_column="target",
                task_type=ExperimentTaskType.REGRESSION,
                compare=ExperimentCompareConfig(optimize="RMSE", n_select=1),
            ),
        )

        bundle = service.compare_models(bundle)

        assert bundle.compare_leaderboard[0].model_name == "Linear Regression"
        assert bundle.compare_leaderboard[0].primary_score == 1.45
        assert bundle.summary.compare_ranking_direction == ExperimentSortDirection.ASCENDING


class TestTuneFlow:
    def test_tune_flow_captures_pre_and_post_metrics(self, classification_df, monkeypatch):
        service = _make_service(monkeypatch, ExperimentTaskType.CLASSIFICATION)
        bundle = service.setup_experiment(
            classification_df,
            ExperimentConfig(
                target_column="target",
                task_type=ExperimentTaskType.CLASSIFICATION,
                compare=ExperimentCompareConfig(optimize="Accuracy", n_select=2),
                tune=ExperimentTuneConfig(optimize="AUC", n_iter=5),
            ),
        )
        bundle = service.compare_models(bundle)

        bundle = service.tune_model(
            bundle,
            ModelSelectionSpec(model_id="lr", model_name="Logistic Regression"),
        )

        assert bundle.tuned_result is not None
        assert bundle.tuned_result.baseline_metrics["Accuracy"] == 0.81
        assert bundle.tuned_result.tuned_metrics["AUC"] == 0.88
        assert bundle.tuned_result.optimize_metric == "AUC"
        assert bundle.tuned_result.applied_config["n_iter"] == 5
        assert bundle.tuned_result.applied_config["optimize"] == "AUC"


class TestPlotHandling:
    def test_plot_artifacts_continue_on_unsupported_plot(self, classification_df, monkeypatch, tmp_path: Path):
        service = PyCaretExperimentService(
            artifacts_dir=tmp_path,
            models_dir=tmp_path / "models",
            snapshots_dir=tmp_path / "snapshots",
            classification_compare_metric="Accuracy",
            regression_compare_metric="R2",
            classification_tune_metric="AUC",
            regression_tune_metric="R2",
            mlflow_experiment_name=None,
        )
        monkeypatch.setattr("app.modeling.pycaret.setup_runner.is_pycaret_available", lambda: True)
        monkeypatch.setattr(
            "app.modeling.pycaret.setup_runner.build_pycaret_experiment",
            lambda resolved_task_type: _FakeClassificationExperiment(),
        )

        bundle = service.setup_experiment(
            classification_df,
            ExperimentConfig(
                target_column="target",
                task_type=ExperimentTaskType.CLASSIFICATION,
                evaluation=ExperimentEvaluationConfig(plots=["confusion_matrix", "unsupported_plot"]),
            ),
            dataset_name="unsafe/dataset:name",
        )
        bundle = service.evaluate_model(
            bundle,
            ModelSelectionSpec(model_id="lr", model_name="Logistic Regression"),
        )

        assert len(bundle.evaluation_plots) == 1
        assert bundle.evaluation_plots[0].plot_id == "confusion_matrix"
        assert any("unsupported_plot" in warning for warning in bundle.warnings)


class TestPersistence:
    def test_finalize_and_save_generates_model_metadata(self, classification_df, monkeypatch, tmp_path: Path):
        service = PyCaretExperimentService(
            artifacts_dir=tmp_path,
            models_dir=tmp_path / "models",
            snapshots_dir=tmp_path / "snapshots",
            classification_compare_metric="Accuracy",
            regression_compare_metric="R2",
            classification_tune_metric="AUC",
            regression_tune_metric="R2",
            mlflow_experiment_name=None,
        )
        monkeypatch.setattr("app.modeling.pycaret.setup_runner.is_pycaret_available", lambda: True)
        monkeypatch.setattr(
            "app.modeling.pycaret.setup_runner.build_pycaret_experiment",
            lambda resolved_task_type: _FakeClassificationExperiment(),
        )

        bundle = service.setup_experiment(
            classification_df,
            ExperimentConfig(
                target_column="target",
                task_type=ExperimentTaskType.CLASSIFICATION,
                persistence=ExperimentPersistenceConfig(save_experiment_snapshot=True),
            ),
            dataset_name="housing",
            dataset_fingerprint="fp-1",
        )
        bundle = service.finalize_and_save_model(
            bundle,
            ModelSelectionSpec(model_id="lr", model_name="Logistic Regression"),
            save_name="best_model",
        )

        assert bundle.saved_model_metadata is not None
        assert bundle.saved_model_metadata.task_type == ExperimentTaskType.CLASSIFICATION
        assert bundle.saved_model_metadata.target_column == "target"
        assert bundle.saved_model_metadata.dataset_fingerprint == "fp-1"
        assert bundle.saved_model_metadata.model_path.exists()
        assert bundle.saved_model_metadata.experiment_snapshot_path is not None
        assert bundle.saved_model_metadata.experiment_snapshot_path.exists()
        assert len(bundle.saved_model_artifacts) == 1
        assert bundle.saved_model_artifacts[0].metadata_path is not None
        assert bundle.saved_model_artifacts[0].metadata_path.exists()

    def test_finalize_and_save_all_models_persists_compared_models(
        self,
        classification_df,
        monkeypatch,
        tmp_path: Path,
    ):
        service = PyCaretExperimentService(
            artifacts_dir=tmp_path,
            models_dir=tmp_path / "models",
            snapshots_dir=tmp_path / "snapshots",
            classification_compare_metric="Accuracy",
            regression_compare_metric="R2",
            classification_tune_metric="AUC",
            regression_tune_metric="R2",
            mlflow_experiment_name=None,
        )
        monkeypatch.setattr("app.modeling.pycaret.setup_runner.is_pycaret_available", lambda: True)
        monkeypatch.setattr(
            "app.modeling.pycaret.setup_runner.build_pycaret_experiment",
            lambda resolved_task_type: _FakeClassificationExperiment(),
        )

        bundle = service.setup_experiment(
            classification_df,
            ExperimentConfig(
                target_column="target",
                task_type=ExperimentTaskType.CLASSIFICATION,
                compare=ExperimentCompareConfig(optimize="Accuracy", n_select=2),
            ),
            dataset_name="housing",
            dataset_fingerprint="fp-1",
        )
        bundle = service.compare_models(bundle)
        bundle = service.finalize_and_save_all_models(bundle, save_name_prefix="auto")

        assert len(bundle.compare_leaderboard) == 2
        assert len(bundle.saved_model_artifacts) == 2
        assert bundle.summary.saved_model_name == "2 model(s) saved for prediction"
        for artifact in bundle.saved_model_artifacts:
            assert artifact.metadata.model_path.exists()
            assert artifact.metadata_path is not None
            assert artifact.metadata_path.exists()
            assert "_saved_model_metadata_" in artifact.metadata_path.name
            assert artifact.metadata.experiment_snapshot_path is None


class TestArtifacts:
    def test_artifact_path_generation(self, tmp_path: Path):
        summary = ExperimentSummary(
            dataset_name="unsafe/folder:name",
            dataset_fingerprint="abc123",
            target_column="target",
            task_type=ExperimentTaskType.CLASSIFICATION,
            execution_backend=ExecutionBackend.LOCAL,
            workspace_mode=WorkspaceMode.DASHBOARD,
            source_row_count=10,
            source_column_count=3,
            feature_column_count=2,
            compare_optimize_metric="Accuracy",
            compare_ranking_direction=ExperimentSortDirection.DESCENDING,
            setup_config=ExperimentSetupConfig(session_id=42).to_summary_model(
                actual_setup_kwargs={"target": "target"}
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

        assert artifacts.summary_json_path is not None
        assert artifacts.summary_json_path.exists()
        assert "/" not in artifacts.summary_json_path.name
        assert ":" not in artifacts.summary_json_path.name
        assert artifacts.summary_json_path == artifacts_second.summary_json_path


class TestMLflowTracking:
    def test_mlflow_wrapper_logs_params_metrics_and_artifacts(self, tmp_path: Path, monkeypatch):
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
        assert fake_mlflow.tracking_uri == "sqlite:///artifacts/mlflow/mlflow.db"
        assert fake_mlflow.registry_uri == "sqlite:///artifacts/mlflow/mlflow.db"
        assert fake_mlflow.experiment_name == "autotabml-experiments"
        assert fake_mlflow.params["task_type"] == "classification"
        assert fake_mlflow.params["tune_optimize_metric"] == "AUC"
        assert fake_mlflow.metrics["best_baseline_score"] == 0.81
        assert fake_mlflow.metrics["tuned_score"] == 0.88
        assert str(summary_artifact) in fake_mlflow.artifacts


class TestPageHelpers:
    def test_build_experiment_run_key_varies_by_target_and_task(self):
        key_a = build_experiment_run_key("housing", "price", ExperimentTaskType.AUTO)
        key_b = build_experiment_run_key("housing", "price", ExperimentTaskType.REGRESSION)
        key_c = build_experiment_run_key("housing", "target", ExperimentTaskType.AUTO)

        assert key_a != key_b
        assert key_a != key_c

    def test_default_metric_helpers(self):
        settings = PyCaretExperimentSettings()

        assert default_compare_metric_for_task(ExperimentTaskType.CLASSIFICATION, settings) == settings.default_compare_metric_classification
        assert default_compare_metric_for_task(ExperimentTaskType.REGRESSION, settings) == settings.default_compare_metric_regression
        assert default_tune_metric_for_task(ExperimentTaskType.CLASSIFICATION, settings) == settings.default_tune_metric_classification
        assert default_tune_metric_for_task(ExperimentTaskType.REGRESSION, settings) == settings.default_tune_metric_regression


class TestConfigDefaults:
    def test_pycaret_settings_defaults(self):
        settings = PyCaretExperimentSettings()

        assert settings.artifacts_dir == Path("artifacts/experiments")
        assert settings.models_dir == Path("artifacts/models")
        assert settings.snapshots_dir == Path("artifacts/experiments/snapshots")
        assert settings.default_compare_metric_classification == "Accuracy"
        assert settings.default_compare_metric_regression == "R2"
        assert settings.default_tune_metric_classification == "AUC"
        assert settings.default_tune_metric_regression == "R2"
        assert settings.default_tracking_mode == "manual"
        assert settings.mlflow_experiment_name == "autotabml-experiments"


class TestCliBoundary:
    def test_experiment_run_cli_loads_dataset_and_invokes_service(self, classification_df, monkeypatch, capsys):
        from app import cli as cli_module

        fake_loaded = type(
            "LoadedDataset",
            (),
            {
                "dataframe": classification_df,
                "metadata": type("Meta", (), {"content_hash": "fp", "schema_hash": None})(),
            },
        )()

        monkeypatch.setattr(cli_module, "_load_cli_dataset", lambda locator, source_type=None: (fake_loaded, "train"))
        monkeypatch.setattr(
            cli_module,
            "load_settings",
            lambda: type(
                "Settings",
                (),
                {
                    "execution": type("Exec", (), {"backend": ExecutionBackend.LOCAL})(),
                    "workspace_mode": WorkspaceMode.DASHBOARD,
                    "pycaret": PyCaretExperimentSettings(),
                    "tracking": type("Tracking", (), {"tracking_uri": None, "registry_uri": None})(),
                },
            )(),
        )

        captured = {}

        class _FakeService:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

            def run_compare_pipeline(self, df, config, **kwargs):
                captured["config"] = config
                return ExperimentResultBundle(
                    dataset_name="train",
                    dataset_fingerprint="fp",
                    config=config,
                    task_type=ExperimentTaskType.CLASSIFICATION,
                    execution_backend=ExecutionBackend.LOCAL,
                    workspace_mode=WorkspaceMode.DASHBOARD,
                    summary=ExperimentSummary(
                        dataset_name="train",
                        dataset_fingerprint="fp",
                        target_column="target",
                        task_type=ExperimentTaskType.CLASSIFICATION,
                        execution_backend=ExecutionBackend.LOCAL,
                        workspace_mode=WorkspaceMode.DASHBOARD,
                        source_row_count=6,
                        source_column_count=3,
                        feature_column_count=2,
                        compare_optimize_metric="Accuracy",
                        compare_ranking_direction=ExperimentSortDirection.DESCENDING,
                        best_baseline_model_name="Logistic Regression",
                        best_baseline_score=0.81,
                        setup_config=ExperimentSetupConfig(session_id=42).to_summary_model(
                            actual_setup_kwargs={"target": "target"}
                        ),
                    ),
                )

        monkeypatch.setattr("app.cli.PyCaretExperimentService", _FakeService)

        args = type(
            "Args",
            (),
            {
                "dataset": "train.csv",
                "target": "target",
                "task_type": "classification",
                "source_type": None,
                "train_size": None,
                "fold": None,
                "fold_strategy": None,
                "preprocess": "auto",
                "ignore_feature": [],
                "compare_metric": None,
                "n_select": 2,
                "turbo": True,
                "budget_time": None,
                "artifacts_dir": None,
            },
        )()

        cli_module.cmd_experiment_run(args)
        output = capsys.readouterr().out

        assert captured["config"].target_column == "target"
        assert captured["config"].compare.n_select == 2
        assert "=== Experiment: train ===" in output
        assert "Best baseline: Logistic Regression" in output

    def test_experiment_run_cli_auto_task_keeps_fold_strategy_unset(self, classification_df, monkeypatch):
        from app import cli as cli_module

        fake_loaded = type(
            "LoadedDataset",
            (),
            {
                "dataframe": classification_df,
                "metadata": type("Meta", (), {"content_hash": "fp", "schema_hash": None})(),
            },
        )()

        monkeypatch.setattr(cli_module, "_load_cli_dataset", lambda locator, source_type=None: (fake_loaded, "train"))
        monkeypatch.setattr(
            cli_module,
            "load_settings",
            lambda: type(
                "Settings",
                (),
                {
                    "execution": type("Exec", (), {"backend": ExecutionBackend.LOCAL})(),
                    "workspace_mode": WorkspaceMode.DASHBOARD,
                    "pycaret": PyCaretExperimentSettings(),
                    "tracking": type("Tracking", (), {"tracking_uri": None, "registry_uri": None})(),
                },
            )(),
        )

        captured = {}

        class _FakeService:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs

            def run_compare_pipeline(self, df, config, **kwargs):
                captured["config"] = config
                return ExperimentResultBundle(
                    dataset_name="train",
                    dataset_fingerprint="fp",
                    config=config,
                    task_type=ExperimentTaskType.CLASSIFICATION,
                    execution_backend=ExecutionBackend.LOCAL,
                    workspace_mode=WorkspaceMode.DASHBOARD,
                    summary=ExperimentSummary(
                        dataset_name="train",
                        dataset_fingerprint="fp",
                        target_column="target",
                        task_type=ExperimentTaskType.CLASSIFICATION,
                        execution_backend=ExecutionBackend.LOCAL,
                        workspace_mode=WorkspaceMode.DASHBOARD,
                        source_row_count=6,
                        source_column_count=3,
                        feature_column_count=2,
                        compare_optimize_metric="Accuracy",
                        compare_ranking_direction=ExperimentSortDirection.DESCENDING,
                        best_baseline_model_name="Logistic Regression",
                        best_baseline_score=0.81,
                        setup_config=ExperimentSetupConfig(session_id=42).to_summary_model(
                            actual_setup_kwargs={"target": "target"}
                        ),
                    ),
                )

        monkeypatch.setattr("app.cli.PyCaretExperimentService", _FakeService)

        args = type(
            "Args",
            (),
            {
                "dataset": "train.csv",
                "target": "target",
                "task_type": "auto",
                "source_type": None,
                "train_size": None,
                "fold": None,
                "fold_strategy": None,
                "preprocess": "auto",
                "ignore_feature": [],
                "compare_metric": None,
                "n_select": 1,
                "turbo": True,
                "budget_time": None,
                "artifacts_dir": None,
            },
        )()

        cli_module.cmd_experiment_run(args)

        assert captured["config"].task_type == ExperimentTaskType.AUTO
        assert captured["config"].setup.fold_strategy is None
        assert captured["config"].tune.optimize is None


def test_build_saved_model_metadata():
    metadata = build_saved_model_metadata(
        task_type=ExperimentTaskType.CLASSIFICATION,
        target_column="target",
        model_id="lr",
        model_name="Logistic Regression",
        model_path=Path("artifacts/models/best.pkl"),
        dataset_fingerprint="fp-1",
        feature_columns=["a", "b"],
        feature_dtypes={"a": "int64", "b": "object"},
        target_dtype="int64",
        experiment_snapshot_path=Path("artifacts/experiments/snapshots/run.pkl"),
    )

    assert metadata.model_id == "lr"
    assert metadata.feature_columns == ["a", "b"]
    assert metadata.experiment_snapshot_includes_data is False