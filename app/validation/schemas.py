"""Pydantic schemas for validation inputs and outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CheckSeverity(str, Enum):
    """Severity level for a single validation check."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class CheckResult(BaseModel):
    """Result of a single validation check."""
    check_name: str
    passed: bool
    severity: CheckSeverity
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    source: str = "app"  # "app" or "gx"


class ValidationRuleConfig(BaseModel):
    """User-facing configuration for a validation run."""
    target_column: str | None = None
    id_columns: list[str] = Field(default_factory=list)
    required_columns: list[str] = Field(default_factory=list)
    min_row_count: int = 1
    null_warn_pct: float = 50.0
    null_fail_pct: float = 95.0
    numeric_range_checks: dict[str, dict[str, float]] = Field(default_factory=dict)
    allowed_category_checks: dict[str, list[str]] = Field(default_factory=dict)
    uniqueness_columns: list[str] = Field(default_factory=list)
    enable_leakage_heuristics: bool = True
    leakage_cardinality_threshold: float = 0.95


class ValidationResultSummary(BaseModel):
    """Aggregated results of a full validation run."""
    run_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_name: str | None = None
    row_count: int = 0
    column_count: int = 0
    total_checks: int = 0
    passed_count: int = 0
    warning_count: int = 0
    failed_count: int = 0
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return self.failed_count > 0


class ValidationArtifactBundle(BaseModel):
    """Paths and metadata for validation artifacts produced."""
    summary_json_path: Path | None = None
    markdown_report_path: Path | None = None
    gx_data_docs_path: Path | None = None
