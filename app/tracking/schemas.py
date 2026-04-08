"""Pydantic schemas for run history, comparison, and MLflow query results."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RunType(str, Enum):
    """Application-level run type classification."""

    BENCHMARK = "benchmark"
    EXPERIMENT = "experiment"
    FLAML = "flaml"
    UNKNOWN = "unknown"


class RunStatus(str, Enum):
    """Normalized MLflow run status."""

    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"
    UNKNOWN = "UNKNOWN"


class ExperimentInfo(BaseModel):
    """Normalized MLflow experiment metadata."""

    experiment_id: str
    name: str
    lifecycle_stage: str = "active"
    artifact_location: str | None = None
    creation_time: datetime | None = None
    last_update_time: datetime | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class RunHistoryItem(BaseModel):
    """Compact normalized summary of a single MLflow run."""

    run_id: str
    experiment_id: str
    experiment_name: str | None = None
    run_name: str | None = None
    status: RunStatus = RunStatus.UNKNOWN
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float | None = None
    artifact_uri: str | None = None
    run_type: RunType = RunType.UNKNOWN
    task_type: str | None = None
    target_column: str | None = None
    dataset_name: str | None = None
    dataset_fingerprint: str | None = None
    model_name: str | None = None
    primary_metric_name: str | None = None
    primary_metric_value: float | None = None
    params: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)


class RunDetailView(RunHistoryItem):
    """Extended run view with full params, metrics, tags, and artifact list."""

    artifact_paths: list[str] = Field(default_factory=list)
    model_info: dict[str, Any] | None = None


class MetricDelta(BaseModel):
    """Side-by-side metric comparison for two runs."""

    name: str
    left_value: float | None = None
    right_value: float | None = None
    delta: float | None = None
    better_side: str | None = None  # "left", "right", "tie", or None


class ConfigDifference(BaseModel):
    """Single configuration key that differs between two runs."""

    key: str
    left_value: str | None = None
    right_value: str | None = None
    category: str = "general"


class ComparisonBundle(BaseModel):
    """Full side-by-side comparison result for two runs."""

    left: RunHistoryItem
    right: RunHistoryItem
    metric_deltas: list[MetricDelta] = Field(default_factory=list)
    config_differences: list[ConfigDifference] = Field(default_factory=list)
    comparable: bool = True
    warnings: list[str] = Field(default_factory=list)
