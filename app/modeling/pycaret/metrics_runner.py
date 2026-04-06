"""Metric catalog inspection and safe custom metric registration."""

from __future__ import annotations

from collections.abc import Callable

from app.modeling.pycaret.errors import PyCaretConfigurationError
from app.modeling.pycaret.schemas import CustomMetricSpec, ExperimentMetricRow, ExperimentTaskType
from app.modeling.pycaret.summary import metric_rows_from_dataframe


def list_available_metrics(experiment_handle) -> list[ExperimentMetricRow]:  # noqa: ANN001
    """Return normalized metric rows from the active experiment."""

    metrics_df = experiment_handle.get_metrics(include_custom=True, raise_errors=False)
    return metric_rows_from_dataframe(metrics_df)


def add_custom_metric(
    experiment_handle,
    *,
    task_type: ExperimentTaskType,
    spec: CustomMetricSpec,
    score_func: Callable,
) -> None:  # noqa: ANN001
    """Register a custom metric on the active experiment."""

    if task_type == ExperimentTaskType.REGRESSION and spec.target != "pred":
        raise PyCaretConfigurationError(
            "Regression custom metrics only support the default target='pred'."
        )

    if task_type == ExperimentTaskType.CLASSIFICATION:
        experiment_handle.add_metric(
            spec.metric_id,
            spec.display_name,
            score_func,
            target=spec.target,
            greater_is_better=spec.greater_is_better,
            multiclass=spec.multiclass,
            **spec.kwargs,
        )
        return

    experiment_handle.add_metric(
        spec.metric_id,
        spec.display_name,
        score_func,
        greater_is_better=spec.greater_is_better,
        **spec.kwargs,
    )


def remove_custom_metric(experiment_handle, name_or_id: str) -> None:  # noqa: ANN001
    """Remove a previously registered custom metric."""

    experiment_handle.remove_metric(name_or_id)