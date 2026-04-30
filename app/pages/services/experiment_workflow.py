"""Page-facing workflow service for the Train & Tune (PyCaret) page."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from app.errors import log_exception
from app.modeling.pycaret.errors import PyCaretExperimentError
from app.modeling.pycaret.schemas import (
    ExperimentCompareConfig,
    ExperimentConfig,
    ExperimentEvaluationConfig,
    ExperimentSetupConfig,
    ExperimentTaskType,
    ExperimentTuneConfig,
    MLflowTrackingMode,
)
from app.observability import Counter, Histogram, correlation_scope, start_span

if TYPE_CHECKING:
    import pandas as pd

    from app.modeling.pycaret.schemas import ExperimentResultBundle
    from app.modeling.pycaret.service import PyCaretExperimentService


logger = logging.getLogger(__name__)

_EXPERIMENT_TRAINING_RUNS_TOTAL = Counter("experiment_training_runs_total")
_EXPERIMENT_TRAINING_FAILURES_TOTAL = Counter("experiment_training_failures_total")
_EXPERIMENT_TRAINING_DURATION_SECONDS = Histogram("experiment_training_duration_seconds")
_EXPERIMENT_AUTOSAVE_ATTEMPTS_TOTAL = Counter("experiment_autosave_attempts_total")
_EXPERIMENT_AUTOSAVE_FAILURES_TOTAL = Counter("experiment_autosave_failures_total")


@dataclass(frozen=True)
class ExperimentFormValues:
    """User-entered form values from the experiment page."""

    target_column: str
    task_type: ExperimentTaskType
    session_id: int
    train_size: float
    fold: int
    preprocess: bool
    ignore_features_raw: str
    use_gpu: Any
    compare_metric: str
    tune_metric: str
    n_select: int
    selected_plots: list[str]
    tracking_mode: MLflowTrackingMode
    enable_log_plots: bool
    enable_log_profile: bool
    enable_log_data: bool


@dataclass(frozen=True)
class TrainingPipelineResult:
    """Outcome from the training+autosave pipeline."""

    bundle: ExperimentResultBundle
    saved_count: int
    autosave_warning: str | None = None


@dataclass(frozen=True)
class TuningInterpretation:
    """Verdict for a tuned-vs-baseline metric comparison."""

    status: str  # "improved", "same", "worse", or "unknown"
    message: str


class ExperimentWorkflowService:
    """Encapsulate Train & Tune page orchestration."""

    def build_run_key(
        self,
        dataset_name: str,
        target_column: str,
        requested_task_type: ExperimentTaskType,
    ) -> str:
        return f"{dataset_name}::{target_column}::{requested_task_type.value}"

    def default_compare_metric(self, task_type: ExperimentTaskType, settings) -> str:  # noqa: ANN001
        if task_type == ExperimentTaskType.CLASSIFICATION:
            return settings.default_compare_metric_classification
        if task_type == ExperimentTaskType.REGRESSION:
            return settings.default_compare_metric_regression
        return ""

    def default_tune_metric(self, task_type: ExperimentTaskType, settings) -> str:  # noqa: ANN001
        if task_type == ExperimentTaskType.CLASSIFICATION:
            return settings.default_tune_metric_classification
        if task_type == ExperimentTaskType.REGRESSION:
            return settings.default_tune_metric_regression
        return ""

    def default_plot_ids(self, task_type: ExperimentTaskType, settings) -> list[str]:  # noqa: ANN001
        if task_type == ExperimentTaskType.CLASSIFICATION:
            return list(settings.default_plot_ids_classification)
        if task_type == ExperimentTaskType.REGRESSION:
            return list(settings.default_plot_ids_regression)
        return []

    def default_tracking_mode(self, settings) -> MLflowTrackingMode:  # noqa: ANN001
        try:
            return MLflowTrackingMode(settings.default_tracking_mode)
        except ValueError:
            return MLflowTrackingMode.MANUAL

    def build_experiment_config(
        self,
        values: ExperimentFormValues,
        settings,  # noqa: ANN001
    ) -> ExperimentConfig:
        ignore_features = [item.strip() for item in values.ignore_features_raw.split(",") if item.strip()]

        if values.task_type == ExperimentTaskType.CLASSIFICATION:
            fold_strategy = settings.default_classification_fold_strategy
        elif values.task_type == ExperimentTaskType.REGRESSION:
            fold_strategy = settings.default_regression_fold_strategy
        else:
            fold_strategy = None

        is_native = values.tracking_mode == MLflowTrackingMode.PYCARET_NATIVE

        return ExperimentConfig(
            target_column=values.target_column,
            task_type=values.task_type,
            mlflow_tracking_mode=values.tracking_mode,
            setup=ExperimentSetupConfig(
                session_id=values.session_id,
                train_size=values.train_size,
                fold=values.fold,
                fold_strategy=fold_strategy,
                ignore_features=ignore_features,
                preprocess=values.preprocess,
                use_gpu=values.use_gpu,
                log_experiment=is_native,
                log_plots=(values.selected_plots if values.enable_log_plots else False),
                log_profile=(values.enable_log_profile if is_native else False),
                log_data=(values.enable_log_data if is_native else False),
            ),
            compare=ExperimentCompareConfig(
                optimize=values.compare_metric or None,
                n_select=values.n_select,
            ),
            tune=ExperimentTuneConfig(optimize=values.tune_metric or None),
            evaluation=ExperimentEvaluationConfig(plots=values.selected_plots),
        )

    def run_training_pipeline(
        self,
        *,
        service: PyCaretExperimentService,
        dataframe: pd.DataFrame,
        config: ExperimentConfig,
        dataset_name: str,
        dataset_fingerprint: str | None,
        execution_backend,  # noqa: ANN001
        workspace_mode,  # noqa: ANN001
        auto_save: bool,
        auto_save_with_snapshots: bool,
    ) -> TrainingPipelineResult:
        task_type_label = config.task_type.value
        tracking_mode_label = config.mlflow_tracking_mode.value
        started_at = perf_counter()
        _EXPERIMENT_TRAINING_RUNS_TOTAL.inc(
            task_type=task_type_label,
            tracking_mode=tracking_mode_label,
        )

        bundle: ExperimentResultBundle | None = None
        autosave_warning: str | None = None
        saved_count = 0

        try:
            with correlation_scope(
                operation="experiment.run_training_pipeline",
                dataset_name=dataset_name,
                dataset_fingerprint=dataset_fingerprint,
                target_column=config.target_column,
                requested_task_type=task_type_label,
            ):
                with start_span(
                    "experiment.run_training_pipeline",
                    dataset_name=dataset_name,
                    target_column=config.target_column,
                    task_type=task_type_label,
                    auto_save=auto_save,
                    tracking_mode=tracking_mode_label,
                ) as span:
                    logger.info(
                        "experiment_training_started",
                        extra={
                            "operation": "experiment.run_training_pipeline",
                            "auto_save": auto_save,
                            "tracking_mode": tracking_mode_label,
                        },
                    )

                    bundle = service.run_compare_pipeline(
                        dataframe,
                        config,
                        dataset_name=dataset_name,
                        dataset_fingerprint=dataset_fingerprint,
                        execution_backend=execution_backend,
                        workspace_mode=workspace_mode,
                    )

                    run_context = {
                        "run_id": getattr(bundle, "mlflow_run_id", None),
                        "experiment_name": getattr(getattr(bundle, "summary", None), "dataset_name", None)
                        or dataset_name,
                        "task_type": getattr(getattr(bundle, "task_type", None), "value", task_type_label),
                    }
                    for key, value in run_context.items():
                        if value is not None:
                            span.set_attribute(key, value)

                    with correlation_scope(**run_context):
                        if auto_save:
                            _EXPERIMENT_AUTOSAVE_ATTEMPTS_TOTAL.inc(
                                task_type=task_type_label,
                                tracking_mode=tracking_mode_label,
                            )
                            try:
                                bundle = service.finalize_and_save_all_models(
                                    bundle,
                                    save_name_prefix="auto",
                                    include_experiment_snapshots=auto_save_with_snapshots,
                                )
                                saved_count = len(bundle.saved_model_artifacts)
                            except PyCaretExperimentError as exc:
                                _EXPERIMENT_AUTOSAVE_FAILURES_TOTAL.inc(
                                    task_type=task_type_label,
                                    tracking_mode=tracking_mode_label,
                                )
                                autosave_warning = str(exc)
                                log_exception(
                                    logger,
                                    exc,
                                    operation="experiment.autosave",
                                    context={
                                        "include_experiment_snapshots": auto_save_with_snapshots,
                                        "save_name_prefix": "auto",
                                    },
                                )

                        logger.info(
                            "experiment_training_completed",
                            extra={
                                "operation": "experiment.run_training_pipeline",
                                "saved_count": saved_count,
                                "auto_save": auto_save,
                                "warning_count": int(autosave_warning is not None),
                            },
                        )
        except Exception as exc:
            _EXPERIMENT_TRAINING_FAILURES_TOTAL.inc(
                task_type=task_type_label,
                tracking_mode=tracking_mode_label,
            )
            _EXPERIMENT_TRAINING_DURATION_SECONDS.observe(
                perf_counter() - started_at,
                task_type=task_type_label,
                tracking_mode=tracking_mode_label,
                status="error",
            )
            log_exception(
                logger,
                exc,
                operation="experiment.run_training_pipeline",
                context={
                    "dataset_name": dataset_name,
                    "target_column": config.target_column,
                    "task_type": task_type_label,
                },
            )
            raise

        _EXPERIMENT_TRAINING_DURATION_SECONDS.observe(
            perf_counter() - started_at,
            task_type=task_type_label,
            tracking_mode=tracking_mode_label,
            status="success",
        )
        if bundle is None:
            raise RuntimeError("Training pipeline completed without producing a bundle.")

        return TrainingPipelineResult(
            bundle=bundle,
            saved_count=saved_count,
            autosave_warning=autosave_warning,
        )

    def interpret_tuning_result(self, tuned_result) -> TuningInterpretation:  # noqa: ANN001
        if tuned_result is None:
            return TuningInterpretation(status="unknown", message="")

        baseline_values = list(tuned_result.baseline_metrics.values())
        tuned_values = list(tuned_result.tuned_metrics.values())
        if not baseline_values or not tuned_values:
            return TuningInterpretation(status="unknown", message="")

        try:
            base = float(baseline_values[0])
            tuned = float(tuned_values[0])
        except (ValueError, TypeError):
            return TuningInterpretation(status="unknown", message="")

        if tuned > base:
            return TuningInterpretation(
                status="improved",
                message="Tuning improved the model. Consider saving the tuned version.",
            )
        if tuned == base:
            return TuningInterpretation(
                status="same",
                message="Tuning produced the same score. The baseline may already be well-optimised.",
            )
        return TuningInterpretation(
            status="worse",
            message="Tuning did not improve the primary score. The baseline version may be the better choice.",
        )


__all__ = [
    "ExperimentFormValues",
    "ExperimentWorkflowService",
    "TrainingPipelineResult",
    "TuningInterpretation",
]
