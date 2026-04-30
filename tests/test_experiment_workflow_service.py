"""Tests for the Train & Tune page workflow service."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from app.modeling.pycaret.errors import PyCaretExperimentError
from app.modeling.pycaret.schemas import (
    ExperimentTaskType,
    MLflowTrackingMode,
)
from app.observability import InMemoryMetricsBackend, install_correlation_filter, set_metrics_backend
from app.pages.services.experiment_workflow import (
    ExperimentFormValues,
    ExperimentWorkflowService,
)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        default_compare_metric_classification="Accuracy",
        default_compare_metric_regression="R2",
        default_tune_metric_classification="F1",
        default_tune_metric_regression="MAE",
        default_plot_ids_classification=["confusion_matrix", "auc"],
        default_plot_ids_regression=["residuals"],
        default_classification_fold_strategy="stratifiedkfold",
        default_regression_fold_strategy="kfold",
        default_tracking_mode="manual",
    )


def _form_values(**overrides: Any) -> ExperimentFormValues:
    base: dict[str, Any] = dict(
        target_column="y",
        task_type=ExperimentTaskType.CLASSIFICATION,
        session_id=123,
        train_size=0.7,
        fold=5,
        preprocess=True,
        ignore_features_raw=" id, name ,, customer_name ",
        use_gpu=False,
        compare_metric="Accuracy",
        tune_metric="F1",
        n_select=3,
        selected_plots=["confusion_matrix"],
        tracking_mode=MLflowTrackingMode.MANUAL,
        enable_log_plots=False,
        enable_log_profile=False,
        enable_log_data=False,
    )
    base.update(overrides)
    return ExperimentFormValues(**base)


class TestExperimentWorkflowService:
    def test_default_metrics_per_task(self) -> None:
        workflow = ExperimentWorkflowService()
        settings = _settings()

        assert workflow.default_compare_metric(ExperimentTaskType.CLASSIFICATION, settings) == "Accuracy"
        assert workflow.default_compare_metric(ExperimentTaskType.REGRESSION, settings) == "R2"
        assert workflow.default_compare_metric(ExperimentTaskType.AUTO, settings) == ""

        assert workflow.default_tune_metric(ExperimentTaskType.CLASSIFICATION, settings) == "F1"
        assert workflow.default_tune_metric(ExperimentTaskType.REGRESSION, settings) == "MAE"

        assert workflow.default_plot_ids(ExperimentTaskType.CLASSIFICATION, settings) == [
            "confusion_matrix",
            "auc",
        ]
        assert workflow.default_plot_ids(ExperimentTaskType.AUTO, settings) == []

    def test_default_tracking_mode_falls_back_to_manual(self) -> None:
        workflow = ExperimentWorkflowService()
        settings = SimpleNamespace(default_tracking_mode="bogus")

        assert workflow.default_tracking_mode(settings) == MLflowTrackingMode.MANUAL

    def test_run_key_is_stable(self) -> None:
        workflow = ExperimentWorkflowService()

        key = workflow.build_run_key("dataset", "y", ExperimentTaskType.CLASSIFICATION)

        assert key == "dataset::y::classification"

    def test_build_experiment_config_parses_ignore_features(self) -> None:
        workflow = ExperimentWorkflowService()

        config = workflow.build_experiment_config(_form_values(), _settings())

        assert config.target_column == "y"
        assert config.task_type == ExperimentTaskType.CLASSIFICATION
        assert config.setup.ignore_features == ["id", "name", "customer_name"]
        assert config.setup.fold_strategy == "stratifiedkfold"
        assert config.setup.log_experiment is False  # Manual tracking mode
        assert config.setup.log_plots is False
        assert config.compare.optimize == "Accuracy"
        assert config.compare.n_select == 3
        assert config.tune.optimize == "F1"
        assert config.evaluation.plots == ["confusion_matrix"]

    def test_build_experiment_config_native_tracking_enables_logging(self) -> None:
        workflow = ExperimentWorkflowService()
        values = _form_values(
            tracking_mode=MLflowTrackingMode.PYCARET_NATIVE,
            enable_log_plots=True,
            enable_log_profile=True,
            enable_log_data=True,
        )

        config = workflow.build_experiment_config(values, _settings())

        assert config.setup.log_experiment is True
        assert config.setup.log_plots == ["confusion_matrix"]
        assert config.setup.log_profile is True
        assert config.setup.log_data is True

    def test_build_experiment_config_regression_fold_strategy(self) -> None:
        workflow = ExperimentWorkflowService()
        values = _form_values(
            task_type=ExperimentTaskType.REGRESSION,
            compare_metric="",
            tune_metric="",
        )

        config = workflow.build_experiment_config(values, _settings())

        assert config.setup.fold_strategy == "kfold"
        assert config.compare.optimize is None
        assert config.tune.optimize is None

    def test_run_training_pipeline_with_autosave_returns_saved_count(self) -> None:
        workflow = ExperimentWorkflowService()

        bundle_after_save = SimpleNamespace(saved_model_artifacts=["a", "b"])

        class _Service:
            def __init__(self) -> None:
                self.compare_calls: list[dict[str, Any]] = []
                self.save_calls: list[dict[str, Any]] = []

            def run_compare_pipeline(self, df, config, **kwargs):  # noqa: ANN001
                self.compare_calls.append({"config": config, **kwargs})
                return SimpleNamespace(saved_model_artifacts=[])

            def finalize_and_save_all_models(self, bundle, **kwargs):  # noqa: ANN001
                self.save_calls.append(kwargs)
                return bundle_after_save

        service = _Service()
        config = workflow.build_experiment_config(_form_values(), _settings())

        result = workflow.run_training_pipeline(
            service=service,
            dataframe=pd.DataFrame({"y": [0, 1]}),
            config=config,
            dataset_name="ds",
            dataset_fingerprint="fp",
            execution_backend="local",
            workspace_mode="local",
            auto_save=True,
            auto_save_with_snapshots=False,
        )

        assert result.bundle is bundle_after_save
        assert result.saved_count == 2
        assert result.autosave_warning is None
        assert service.save_calls[0]["save_name_prefix"] == "auto"
        assert service.save_calls[0]["include_experiment_snapshots"] is False

    def test_run_training_pipeline_captures_autosave_failure(self) -> None:
        workflow = ExperimentWorkflowService()
        compare_bundle = SimpleNamespace(saved_model_artifacts=[])

        class _Service:
            def run_compare_pipeline(self, df, config, **kwargs):  # noqa: ANN001
                return compare_bundle

            def finalize_and_save_all_models(self, bundle, **kwargs):  # noqa: ANN001
                raise PyCaretExperimentError("disk full")

        config = workflow.build_experiment_config(_form_values(), _settings())

        result = workflow.run_training_pipeline(
            service=_Service(),
            dataframe=pd.DataFrame({"y": [0, 1]}),
            config=config,
            dataset_name="ds",
            dataset_fingerprint=None,
            execution_backend="local",
            workspace_mode="local",
            auto_save=True,
            auto_save_with_snapshots=False,
        )

        assert result.bundle is compare_bundle
        assert result.saved_count == 0
        assert result.autosave_warning is not None
        assert "disk full" in result.autosave_warning

    def test_run_training_pipeline_emits_observability(self, caplog: pytest.LogCaptureFixture) -> None:
        workflow = ExperimentWorkflowService()
        bundle_after_save = SimpleNamespace(
            saved_model_artifacts=["a"],
            mlflow_run_id="run-123",
            task_type=ExperimentTaskType.CLASSIFICATION,
            summary=SimpleNamespace(dataset_name="ds"),
        )

        class _Service:
            def run_compare_pipeline(self, df, config, **kwargs):  # noqa: ANN001, ARG002
                return bundle_after_save

            def finalize_and_save_all_models(self, bundle, **kwargs):  # noqa: ANN001, ARG002
                return bundle_after_save

        backend = InMemoryMetricsBackend()
        previous_backend = set_metrics_backend(backend)
        caplog.set_level(logging.INFO, logger="app.pages.services.experiment_workflow")
        install_correlation_filter(logging.getLogger())
        try:
            workflow.run_training_pipeline(
                service=_Service(),
                dataframe=pd.DataFrame({"y": [0, 1]}),
                config=workflow.build_experiment_config(_form_values(), _settings()),
                dataset_name="ds",
                dataset_fingerprint="fp",
                execution_backend="local",
                workspace_mode="local",
                auto_save=True,
                auto_save_with_snapshots=False,
            )
        finally:
            set_metrics_backend(previous_backend)

        completed = [record for record in caplog.records if record.getMessage() == "experiment_training_completed"]
        assert completed
        assert getattr(completed[-1], "run_id", None) == "run-123"
        assert any(
            key[0] == "experiment_training_runs_total" and dict(key[1])["task_type"] == "classification"
            for key in backend.counters
        )
        assert any(
            key[0] == "experiment_training_duration_seconds" and dict(key[1])["status"] == "success"
            for key in backend.histograms
        )

    def test_interpret_tuning_result_states(self) -> None:
        workflow = ExperimentWorkflowService()

        improved = workflow.interpret_tuning_result(
            SimpleNamespace(baseline_metrics={"Accuracy": 0.7}, tuned_metrics={"Accuracy": 0.8})
        )
        assert improved.status == "improved"

        same = workflow.interpret_tuning_result(
            SimpleNamespace(baseline_metrics={"Accuracy": 0.7}, tuned_metrics={"Accuracy": 0.7})
        )
        assert same.status == "same"

        worse = workflow.interpret_tuning_result(
            SimpleNamespace(baseline_metrics={"Accuracy": 0.8}, tuned_metrics={"Accuracy": 0.6})
        )
        assert worse.status == "worse"

        none_result = workflow.interpret_tuning_result(None)
        assert none_result.status == "unknown"

        non_numeric = workflow.interpret_tuning_result(
            SimpleNamespace(baseline_metrics={"k": "n/a"}, tuned_metrics={"k": "n/a"})
        )
        assert non_numeric.status == "unknown"
