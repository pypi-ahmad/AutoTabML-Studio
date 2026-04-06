"""Pydantic schemas for profiling inputs and outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.config.models import ProfilingMode


class ProfilingConfig(BaseModel):
    """User-facing configuration for a profiling run."""
    mode: ProfilingMode = ProfilingMode.STANDARD
    large_dataset_row_threshold: int = 50_000
    large_dataset_col_threshold: int = 100
    sampling_row_threshold: int = 200_000
    sample_size: int = 50_000
    title: str | None = None


class ProfilingResultSummary(BaseModel):
    """Quick-access summary extracted from a profiling run."""
    run_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_name: str | None = None
    row_count: int = 0
    column_count: int = 0
    numeric_column_count: int = 0
    categorical_column_count: int = 0
    missing_cells_total: int = 0
    missing_cells_pct: float = 0.0
    duplicate_row_count: int = 0
    duplicate_row_pct: float = 0.0
    memory_bytes: int | None = None
    report_mode: ProfilingMode = ProfilingMode.STANDARD
    sampling_applied: bool = False
    sample_size_used: int | None = None
    high_cardinality_columns: list[str] = Field(default_factory=list)
    target_imbalance_hint: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ProfilingArtifactBundle(BaseModel):
    """Paths and metadata for profiling artifacts produced."""
    html_report_path: Path | None = None
    summary_json_path: Path | None = None
