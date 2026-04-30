"""Pydantic schemas for benchmark configuration and results."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.config.enums import ExecutionBackend, WorkspaceMode


class BenchmarkTaskType(str, Enum):
    """Supported benchmark tasks."""

    AUTO = "auto"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class BenchmarkSortDirection(str, Enum):
    """Metric ordering direction for ranking."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


class BenchmarkSplitConfig(BaseModel):
    """Explicit train/test split configuration."""

    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = 42
    stratify: bool | None = None


class BenchmarkConfig(BaseModel):
    """User-facing benchmark execution configuration."""

    target_column: str
    task_type: BenchmarkTaskType = BenchmarkTaskType.AUTO
    prefer_gpu: bool = True
    split: BenchmarkSplitConfig = Field(default_factory=BenchmarkSplitConfig)
    ranking_metric: str | None = None
    sample_rows: int | None = Field(default=None, gt=0)
    include_models: list[str] = Field(default_factory=list)
    exclude_models: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, gt=0)
    categorical_encoder: str = "onehot"
    ignore_warnings: bool = True
    max_models: int | None = Field(default=None, gt=0)
    timeout_seconds: float | None = Field(default=None, gt=0)


class BenchmarkResultRow(BaseModel):
    """Normalized single-model benchmark result row."""

    model_name: str
    task_type: BenchmarkTaskType
    primary_score: float | None = None
    raw_metrics: dict[str, Any] = Field(default_factory=dict)
    training_time_seconds: float | None = None
    rank: int | None = None
    run_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    benchmark_backend: ExecutionBackend = ExecutionBackend.LOCAL
    warnings: list[str] = Field(default_factory=list)


class BenchmarkSummary(BaseModel):
    """Roll-up summary for a benchmark run."""

    run_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_name: str | None = None
    dataset_fingerprint: str | None = None
    target_column: str
    task_type: BenchmarkTaskType
    benchmark_backend: ExecutionBackend = ExecutionBackend.LOCAL
    workspace_mode: WorkspaceMode | None = None
    ranking_metric: str
    ranking_direction: BenchmarkSortDirection
    source_row_count: int
    source_column_count: int
    benchmark_row_count: int
    feature_column_count: int
    train_row_count: int
    test_row_count: int
    sampled_row_count: int | None = None
    stratified_split_applied: bool = False
    model_count: int = 0
    best_model_name: str | None = None
    best_score: float | None = None
    fastest_model_name: str | None = None
    fastest_model_time_seconds: float | None = None
    benchmark_duration_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    split_config: BenchmarkSplitConfig


class BenchmarkArtifactBundle(BaseModel):
    """Artifact paths emitted for a benchmark run."""

    raw_results_csv_path: Path | None = None
    leaderboard_csv_path: Path | None = None
    leaderboard_json_path: Path | None = None
    summary_json_path: Path | None = None
    markdown_summary_path: Path | None = None
    score_chart_path: Path | None = None
    training_time_chart_path: Path | None = None


class BenchmarkSavedModelMetadata(BaseModel):
    """Metadata sidecar for a benchmark-saved local model."""

    source: str = "benchmark"
    model_name: str
    task_type: BenchmarkTaskType
    target_column: str
    dataset_name: str | None = None
    dataset_fingerprint: str | None = None
    trained_at: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    split_test_size: float | None = None
    split_random_state: int | None = None
    model_path: Path
    artifact_format: str | None = None
    trusted_source: str | None = None
    model_sha256: str | None = None
    trusted_types: list[str] = Field(default_factory=list)


class BenchmarkResultBundle(BaseModel):
    """Complete benchmark result package for UI, CLI, and tracking."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_name: str | None = None
    dataset_fingerprint: str | None = None
    config: BenchmarkConfig
    task_type: BenchmarkTaskType
    benchmark_backend: ExecutionBackend = ExecutionBackend.LOCAL
    workspace_mode: WorkspaceMode | None = None
    raw_results: pd.DataFrame
    leaderboard: list[BenchmarkResultRow] = Field(default_factory=list)
    top_models: list[BenchmarkResultRow] = Field(default_factory=list)
    summary: BenchmarkSummary
    artifacts: BenchmarkArtifactBundle | None = None
    mlflow_run_id: str | None = None
    warnings: list[str] = Field(default_factory=list)