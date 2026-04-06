"""Build a unified ValidationResultSummary from check results."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.validation.schemas import (
    CheckResult,
    CheckSeverity,
    ValidationResultSummary,
)


def build_summary(
    checks: list[CheckResult],
    df: pd.DataFrame,
    *,
    dataset_name: str | None = None,
) -> ValidationResultSummary:
    """Aggregate individual check results into a summary."""
    passed = sum(1 for c in checks if c.passed)
    warnings = sum(
        1 for c in checks if not c.passed and c.severity == CheckSeverity.WARNING
    )
    failed = sum(
        1 for c in checks if not c.passed and c.severity == CheckSeverity.ERROR
    )

    return ValidationResultSummary(
        run_timestamp=datetime.now(timezone.utc),
        dataset_name=dataset_name,
        row_count=len(df),
        column_count=len(df.columns),
        total_checks=len(checks),
        passed_count=passed,
        warning_count=warnings,
        failed_count=failed,
        checks=checks,
    )
