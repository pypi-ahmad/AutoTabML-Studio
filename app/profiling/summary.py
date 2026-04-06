"""Extract structured summary for profiling runs.

Core quick-access metrics are intentionally derived from the original
DataFrame instead of ydata-profiling internals. This keeps summary fields
stable across library versions and ensures sampled HTML reports do not leak
sample-only counts into dataset-level summary cards.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.config.models import ProfilingMode
from app.profiling.schemas import ProfilingResultSummary


def extract_summary(
    _report: Any,
    original_df: pd.DataFrame,
    *,
    effective_mode: ProfilingMode,
    was_sampled: bool,
    sample_size_used: int | None,
    dataset_name: str | None,
) -> ProfilingResultSummary:
    """Build a ProfilingResultSummary from a ydata-profiling report object.

    The HTML profile itself may be generated from a sampled DataFrame for
    safety, but the quick summary shown in the app and CLI always describes
    the full dataset supplied to this function.
    """
    n_rows, n_cols = original_df.shape

    missing_total = int(original_df.isna().sum().sum())
    total_cells = n_rows * n_cols
    missing_pct = (missing_total / total_cells * 100) if total_cells > 0 else 0.0

    dup_count = int(original_df.duplicated().sum())
    dup_pct = (dup_count / n_rows * 100) if n_rows > 0 else 0.0

    numeric_count = int(original_df.select_dtypes(include="number").shape[1])
    categorical_count = int(
        original_df.select_dtypes(include=["object", "string", "category"]).shape[1]
    )
    high_card = _get_high_cardinality_columns(original_df)
    mem = int(original_df.memory_usage(deep=True).sum())

    return ProfilingResultSummary(
        run_timestamp=datetime.now(timezone.utc),
        dataset_name=dataset_name,
        row_count=n_rows,
        column_count=n_cols,
        numeric_column_count=numeric_count,
        categorical_column_count=categorical_count,
        missing_cells_total=missing_total,
        missing_cells_pct=round(missing_pct, 2),
        duplicate_row_count=dup_count,
        duplicate_row_pct=round(dup_pct, 2),
        memory_bytes=mem,
        report_mode=effective_mode,
        sampling_applied=was_sampled,
        sample_size_used=sample_size_used,
        high_cardinality_columns=high_card,
    )


def _get_high_cardinality_columns(
    df: pd.DataFrame, threshold: float = 0.9
) -> list[str]:
    """Identify columns with very high cardinality relative to row count."""
    high_card: list[str] = []
    n_rows = len(df)
    if n_rows == 0:
        return high_card
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        ratio = df[col].nunique() / n_rows
        if ratio >= threshold:
            high_card.append(str(col))
    return high_card
