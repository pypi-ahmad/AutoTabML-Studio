"""Task routing, metric ordering, and plot selection helpers."""

from __future__ import annotations

import pandas as pd

from app.modeling.benchmark.schemas import BenchmarkTaskType
from app.modeling.benchmark.selectors import infer_task_type as benchmark_infer_task_type
from app.modeling.benchmark.selectors import validate_target as benchmark_validate_target
from app.modeling.pycaret.errors import UnsupportedExperimentTaskError
from app.modeling.pycaret.schemas import ExperimentSortDirection, ExperimentTaskType, ModelSelectionSpec

SUPPORTED_CLASSIFICATION_PLOTS = [
    "confusion_matrix",
    "auc",
    "pr",
    "class_report",
    "calibration",
    "feature",
]

SUPPORTED_REGRESSION_PLOTS = [
    "residuals",
    "error",
    "feature",
]

LOWER_IS_BETTER_MARKERS = ("rmse", "mae", "mse", "rmsle", "mape", "loss", "error")


def resolve_task_type(
    target: pd.Series,
    requested_task_type: ExperimentTaskType,
) -> tuple[ExperimentTaskType, list[str]]:
    """Resolve the effective experiment task type and validate the target."""

    warnings: list[str] = []
    non_null_target = target.dropna()
    if requested_task_type == ExperimentTaskType.AUTO:
        inferred = benchmark_infer_task_type(non_null_target)
        task_type = _map_from_benchmark_task_type(inferred)
        warnings.append(f"Task type auto-detected as {task_type.value}.")
    else:
        task_type = requested_task_type

    warnings.extend(
        benchmark_validate_target(
            non_null_target,
            _map_to_benchmark_task_type(task_type),
        )
    )
    return task_type, warnings


def metric_sort_direction(metric_name: str) -> ExperimentSortDirection:
    """Return the expected ordering direction for a metric name."""

    lowered = metric_name.lower()
    if any(marker in lowered for marker in LOWER_IS_BETTER_MARKERS):
        return ExperimentSortDirection.ASCENDING
    return ExperimentSortDirection.DESCENDING


def supported_plots_for_task(task_type: ExperimentTaskType) -> list[str]:
    """Return the supported plot ids for the given task."""

    if task_type == ExperimentTaskType.CLASSIFICATION:
        return list(SUPPORTED_CLASSIFICATION_PLOTS)
    if task_type == ExperimentTaskType.REGRESSION:
        return list(SUPPORTED_REGRESSION_PLOTS)
    raise UnsupportedExperimentTaskError(f"Unsupported experiment task type: {task_type.value}.")


def resolve_model_id(selection: ModelSelectionSpec, model_name_to_id: dict[str, str]) -> str | None:
    """Resolve a model id from explicit or name-based selection."""

    if selection.model_id:
        return selection.model_id
    return model_name_to_id.get(selection.model_name)


def _map_to_benchmark_task_type(task_type: ExperimentTaskType) -> BenchmarkTaskType:
    if task_type == ExperimentTaskType.CLASSIFICATION:
        return BenchmarkTaskType.CLASSIFICATION
    if task_type == ExperimentTaskType.REGRESSION:
        return BenchmarkTaskType.REGRESSION
    return BenchmarkTaskType.AUTO


def _map_from_benchmark_task_type(task_type: BenchmarkTaskType) -> ExperimentTaskType:
    if task_type == BenchmarkTaskType.CLASSIFICATION:
        return ExperimentTaskType.CLASSIFICATION
    if task_type == BenchmarkTaskType.REGRESSION:
        return ExperimentTaskType.REGRESSION
    return ExperimentTaskType.AUTO