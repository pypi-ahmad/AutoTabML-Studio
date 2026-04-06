"""Ranking logic for normalized benchmark results."""

from __future__ import annotations

from math import inf
from collections.abc import Iterable

import pandas as pd

from app.modeling.benchmark.errors import BenchmarkExecutionError
from app.modeling.benchmark.schemas import (
    BenchmarkResultRow,
    BenchmarkSortDirection,
    BenchmarkTaskType,
)

DEFAULT_CLASSIFICATION_METRICS = [
    "Balanced Accuracy",
    "F1 Score",
    "Accuracy",
    "ROC AUC",
    "Precision",
    "Recall",
]

DEFAULT_REGRESSION_METRICS = [
    "Adjusted R-Squared",
    "R-Squared",
    "RMSE",
]

LOWER_IS_BETTER_MARKERS = ("rmse", "mae", "mse", "log loss", "loss", "error")


def resolve_ranking_metric(
    task_type: BenchmarkTaskType,
    available_metrics: Iterable[str],
    *,
    preferred_metric: str | None = None,
    default_metric: str | None = None,
    raw_results: pd.DataFrame | None = None,
) -> tuple[str, BenchmarkSortDirection, list[str]]:
    """Resolve a safe ranking metric and its ordering direction."""

    warnings: list[str] = []
    available = list(dict.fromkeys(metric for metric in available_metrics if metric))
    if not available:
        raise BenchmarkExecutionError("Benchmark results do not contain any usable metric columns.")

    candidates: list[str] = []
    if preferred_metric:
        candidates.append(preferred_metric)
    if default_metric and default_metric not in candidates:
        candidates.append(default_metric)

    task_defaults = (
        DEFAULT_CLASSIFICATION_METRICS
        if task_type == BenchmarkTaskType.CLASSIFICATION
        else DEFAULT_REGRESSION_METRICS
    )
    for metric in task_defaults:
        if metric not in candidates:
            candidates.append(metric)

    fallback_candidates = [metric for metric in available if metric != "Time Taken"]
    for metric in fallback_candidates:
        if metric not in candidates:
            candidates.append(metric)

    for metric in candidates:
        if metric in available:
            if raw_results is not None and _is_degenerate_r2(metric, raw_results):
                warnings.append(
                    f"Ranking metric '{metric}' has degenerate values (> 1.0), "
                    "likely due to very few test samples relative to features; "
                    "falling back to next available metric."
                )
                continue
            if preferred_metric and metric != preferred_metric:
                warnings.append(
                    f"Ranking metric '{preferred_metric}' was unavailable; fell back to '{metric}'."
                )
            return metric, metric_sort_direction(metric), warnings

    raise BenchmarkExecutionError("Unable to resolve a usable ranking metric from benchmark results.")


def _is_degenerate_r2(metric_name: str, raw_results: pd.DataFrame) -> bool:
    """Return True if an R²-family metric has degenerate values (> 1.0).

    This happens when the number of test samples is too small relative to
    the number of features, causing the Adjusted R-Squared formula denominator
    ``(n - p - 1)`` to become zero or negative.
    """
    lowered = metric_name.lower()
    if "r-squared" not in lowered and "r2" not in lowered:
        return False
    if metric_name not in raw_results.columns:
        return False
    values = pd.to_numeric(raw_results[metric_name], errors="coerce").dropna()
    if values.empty:
        return False
    return bool(values.max() > 1.0)


def metric_sort_direction(metric_name: str) -> BenchmarkSortDirection:
    """Return the expected sort direction for a metric name."""

    lowered = metric_name.lower()
    if any(marker in lowered for marker in LOWER_IS_BETTER_MARKERS):
        return BenchmarkSortDirection.ASCENDING
    return BenchmarkSortDirection.DESCENDING


def rank_result_rows(
    rows: list[BenchmarkResultRow],
    *,
    ranking_metric: str,
    direction: BenchmarkSortDirection,
) -> list[BenchmarkResultRow]:
    """Return a ranked copy of the normalized result rows."""

    ranked_rows = [row.model_copy(deep=True) for row in rows]

    def metric_value(row: BenchmarkResultRow) -> float | None:
        value = row.raw_metrics.get(ranking_metric)
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def training_time_value(row: BenchmarkResultRow) -> float:
        return row.training_time_seconds if row.training_time_seconds is not None else inf

    if direction == BenchmarkSortDirection.DESCENDING:
        ranked_rows.sort(
            key=lambda row: (
                metric_value(row) is None,
                -(metric_value(row) or 0.0),
                training_time_value(row),
                row.model_name,
            )
        )
    else:
        ranked_rows.sort(
            key=lambda row: (
                metric_value(row) is None,
                metric_value(row) if metric_value(row) is not None else inf,
                training_time_value(row),
                row.model_name,
            )
        )

    for index, row in enumerate(ranked_rows, start=1):
        row.rank = index
        row.primary_score = metric_value(row)

    return ranked_rows