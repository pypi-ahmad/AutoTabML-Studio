"""Filtering and sorting helpers for run history queries."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.tracking.schemas import RunStatus, RunType


class RunSortField(str, Enum):
    """Fields available for sorting run history."""

    START_TIME = "start_time"
    DURATION = "duration"
    MODEL_NAME = "model_name"
    PRIMARY_SCORE = "primary_score"


class SortDirection(str, Enum):
    """Ascending or descending sort."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


class RunHistoryFilter(BaseModel):
    """Declarative filter for run history queries."""

    experiment_names: list[str] = Field(default_factory=list)
    run_type: RunType | None = None
    task_type: str | None = None
    dataset_fingerprint: str | None = None
    target_column: str | None = None
    model_name: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    status: RunStatus | None = None


class RunHistorySort(BaseModel):
    """Sort specification for run history queries."""

    field: RunSortField = RunSortField.START_TIME
    direction: SortDirection = SortDirection.DESCENDING


def build_mlflow_filter_string(history_filter: RunHistoryFilter | None) -> str:
    """Build an MLflow-compatible filter string from a RunHistoryFilter."""

    if history_filter is None:
        return ""

    clauses: list[str] = []

    if history_filter.status is not None:
        clauses.append(f"attributes.status = '{history_filter.status.value}'")

    if history_filter.task_type:
        clauses.append(f"params.task_type = '{history_filter.task_type}'")

    if history_filter.dataset_fingerprint:
        clauses.append(f"params.dataset_fingerprint = '{history_filter.dataset_fingerprint}'")

    if history_filter.target_column:
        clauses.append(f"params.target_column = '{history_filter.target_column}'")

    for key, value in history_filter.tags.items():
        safe_key = key.replace("'", "")
        safe_value = value.replace("'", "")
        clauses.append(f"tags.`{safe_key}` = '{safe_value}'")

    return " AND ".join(clauses)
