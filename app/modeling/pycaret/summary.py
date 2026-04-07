"""Normalization and summary helpers for experiment outputs."""

from __future__ import annotations

import math
from os import PathLike
from typing import Any

import pandas as pd

from app.modeling.pycaret.schemas import ExperimentLeaderboardRow, ExperimentMetricRow, ExperimentSortDirection
from app.modeling.pycaret.selectors import metric_sort_direction

MODEL_NAME_COLUMNS = ("Model Name", "Model", "Estimator")
MODEL_ID_COLUMNS = ("ID", "Model ID", "model_id")
NAME_COLUMNS = ("Display Name", "Name")
GREATER_COLUMNS = ("Greater is Better", "greater_is_better")
CUSTOM_COLUMNS = ("Custom", "custom")


def safe_json_value(value: Any) -> Any:
    """Return a JSON-safe representation of a value."""

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, PathLike):
        return str(value)
    if isinstance(value, dict):
        return {str(key): safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def sanitize_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe mapping."""

    return {str(key): safe_json_value(value) for key, value in mapping.items()}


def leaderboard_to_dataframe(rows: list[ExperimentLeaderboardRow]) -> pd.DataFrame:
    """Convert normalized leaderboard rows to a flat dataframe."""

    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {
            "Rank": row.rank,
            "ID": row.model_id,
            "Model": row.model_name,
            "Score": row.primary_score,
            "Stage": row.stage,
            "Warnings": "; ".join(row.warnings),
        }
        for key, value in row.raw_metrics.items():
            record[key] = value
        records.append(record)
    return pd.DataFrame(records)


def metric_rows_from_dataframe(metrics_df: pd.DataFrame) -> list[ExperimentMetricRow]:
    """Normalize the metric catalog dataframe returned by PyCaret."""

    rows: list[ExperimentMetricRow] = []
    if metrics_df.empty:
        return rows

    for index, values in metrics_df.iterrows():
        raw = sanitize_mapping(values.to_dict())
        metric_id = str(index)
        display_name = str(_first_present(raw, NAME_COLUMNS, default=metric_id))
        greater_raw = _first_present(raw, GREATER_COLUMNS)
        rows.append(
            ExperimentMetricRow(
                metric_id=metric_id,
                display_name=display_name,
                greater_is_better=_coerce_bool(greater_raw),
                is_custom=bool(_coerce_bool(_first_present(raw, CUSTOM_COLUMNS), default=False)),
                raw_values=raw,
            )
        )
    return rows


def normalize_compare_grid(
    score_grid: pd.DataFrame,
    *,
    requested_metric: str | None,
    model_name_to_id: dict[str, str],
) -> tuple[list[ExperimentLeaderboardRow], str | None, ExperimentSortDirection | None, list[str]]:
    """Normalize the compare_models score grid into stable rows."""

    warnings: list[str] = []
    if score_grid.empty:
        return [], requested_metric, None, ["PyCaret compare_models returned an empty leaderboard."]

    name_column = _first_existing_column(score_grid, MODEL_NAME_COLUMNS)
    available_metrics = [
        column
        for column in score_grid.columns
        if column not in set(MODEL_NAME_COLUMNS) | set(MODEL_ID_COLUMNS)
    ]
    resolved_metric = resolve_metric_name(requested_metric, available_metrics)
    if requested_metric and resolved_metric != requested_metric:
        warnings.append(
            f"Compare metric '{requested_metric}' was unavailable; fell back to '{resolved_metric}'."
        )
    direction = metric_sort_direction(resolved_metric) if resolved_metric is not None else None

    rows: list[ExperimentLeaderboardRow] = []
    for index, values in score_grid.iterrows():
        raw = sanitize_mapping(values.to_dict())
        model_name = str(raw.get(name_column) or index)
        rows.append(
            ExperimentLeaderboardRow(
                stage="compare",
                model_id=model_name_to_id.get(model_name) or _first_present(raw, MODEL_ID_COLUMNS),
                model_name=model_name,
                primary_score=coerce_float(raw.get(resolved_metric)) if resolved_metric else None,
                raw_metrics=raw,
            )
        )

    rows = rank_leaderboard_rows(rows, direction=direction, metric_name=resolved_metric)
    return rows, resolved_metric, direction, warnings


def extract_mean_metrics(score_grid: pd.DataFrame) -> dict[str, Any]:
    """Extract stable mean metrics from a create_model/tune_model score grid."""

    if score_grid.empty:
        return {}

    mean_row = None
    for index, values in score_grid.iterrows():
        if str(index).strip().lower() == "mean":
            mean_row = values
            break

    if mean_row is None:
        fold_column = _first_existing_column(score_grid, ("Fold", "fold"))
        if fold_column is not None:
            mean_candidates = score_grid.loc[
                score_grid[fold_column].astype(str).str.strip().str.lower() == "mean"
            ]
            if not mean_candidates.empty:
                mean_row = mean_candidates.iloc[0]

    if mean_row is None:
        mean_row = score_grid.select_dtypes(include=["number"]).mean(numeric_only=True)

    if isinstance(mean_row, pd.Series):
        return sanitize_mapping(mean_row.to_dict())
    return {}


def rank_leaderboard_rows(
    rows: list[ExperimentLeaderboardRow],
    *,
    direction: ExperimentSortDirection | None,
    metric_name: str | None,
) -> list[ExperimentLeaderboardRow]:
    """Return a ranked copy of normalized leaderboard rows."""

    ranked_rows = [row.model_copy(deep=True) for row in rows]
    if direction is None or metric_name is None:
        for index, row in enumerate(ranked_rows, start=1):
            row.rank = index
        return ranked_rows

    def metric_value(row: ExperimentLeaderboardRow) -> float | None:
        return coerce_float(row.raw_metrics.get(metric_name))

    if direction == ExperimentSortDirection.DESCENDING:
        ranked_rows.sort(
            key=lambda row: (
                metric_value(row) is None,
                -(metric_value(row) or 0.0),
                row.model_name,
            )
        )
    else:
        ranked_rows.sort(
            key=lambda row: (
                metric_value(row) is None,
                metric_value(row) if metric_value(row) is not None else math.inf,
                row.model_name,
            )
        )

    for index, row in enumerate(ranked_rows, start=1):
        row.rank = index
        row.primary_score = metric_value(row)

    return ranked_rows


def resolve_metric_name(preferred_metric: str | None, available_columns: list[str]) -> str | None:
    """Resolve a usable metric name from the available columns."""

    if not available_columns:
        return None
    if preferred_metric and preferred_metric in available_columns:
        return preferred_metric

    filtered = [column for column in available_columns if column.lower() not in {"tt (sec)", "time taken"}]
    if filtered:
        return filtered[0]
    return available_columns[0]


def coerce_float(value: Any) -> float | None:
    """Best-effort float conversion with null handling."""

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _first_present(mapping: dict[str, Any], candidates: tuple[str, ...], *, default: Any = None) -> Any:
    for candidate in candidates:
        if candidate in mapping and mapping[candidate] is not None:
            return mapping[candidate]
    return default


def _coerce_bool(value: Any, *, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)