"""Typed records for the local app metadata database."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AppJobType(str, Enum):
    """Local app job categories stored outside MLflow."""

    VALIDATION = "validation"
    PROFILING = "profiling"
    BENCHMARK = "benchmark"
    EXPERIMENT = "experiment"
    PREDICTION = "prediction"


class AppJobStatus(str, Enum):
    """Coarse local execution status."""

    SUCCESS = "success"
    FAILED = "failed"


class ProjectRecord(BaseModel):
    """Local project metadata for workspace navigation convenience."""

    project_id: str
    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DatasetRecord(BaseModel):
    """Locally persisted dataset lineage record."""

    dataset_key: str
    project_id: str | None = None
    display_name: str | None = None
    source_type: str
    source_locator: str
    schema_hash: str
    content_hash: str | None = None
    row_count: int = 0
    column_count: int = 0
    column_names: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobRecord(BaseModel):
    """Local app job record used for dashboard/history convenience."""

    job_id: str
    job_type: AppJobType
    status: AppJobStatus
    project_id: str | None = None
    dataset_key: str | None = None
    dataset_name: str | None = None
    title: str | None = None
    mlflow_run_id: str | None = None
    primary_artifact_path: Path | None = None
    summary_path: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SavedLocalModelRecord(BaseModel):
    """Local saved-model metadata kept outside MLflow registry state."""

    record_id: str
    model_name: str
    model_path: Path
    task_type: str
    target_column: str | None = None
    dataset_fingerprint: str | None = None
    metadata_path: Path | None = None
    experiment_snapshot_path: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchRunStatus(str, Enum):
    """Overall batch run status."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class BatchItemStatus(str, Enum):
    """Individual dataset run status within a batch."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class BatchRunRecord(BaseModel):
    """Tracks an overall batch execution run."""

    batch_id: str
    batch_name: str
    total_datasets: int = 0
    completed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    status: BatchRunStatus = BatchRunStatus.RUNNING
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchRunItemRecord(BaseModel):
    """Tracks a single dataset within a batch run through validate/profile/benchmark."""

    item_id: str
    batch_id: str
    uci_id: int
    dataset_name: str
    target_column: str | None = None
    task_type: str | None = None
    row_count: int | None = None
    column_count: int | None = None
    status: BatchItemStatus = BatchItemStatus.PENDING
    validation_status: str | None = None
    profiling_status: str | None = None
    benchmark_status: str | None = None
    best_model: str | None = None
    best_score: float | None = None
    ranking_metric: str | None = None
    mlflow_run_id: str | None = None
    duration_seconds: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))