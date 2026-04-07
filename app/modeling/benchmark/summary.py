"""Normalization and summary helpers for benchmark results."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.config.enums import ExecutionBackend, WorkspaceMode
from app.modeling.benchmark.schemas import (
    BenchmarkConfig,
    BenchmarkResultRow,
    BenchmarkSortDirection,
    BenchmarkSummary,
    BenchmarkTaskType,
)


def build_result_rows(
    raw_results: pd.DataFrame,
    *,
    task_type: BenchmarkTaskType,
    benchmark_backend: ExecutionBackend,
    run_timestamp: datetime | None = None,
) -> list[BenchmarkResultRow]:
    """Normalize a raw LazyPredict scores dataframe into result rows."""

    timestamp = run_timestamp or datetime.now(timezone.utc)
    rows: list[BenchmarkResultRow] = []

    if raw_results.empty:
        return rows

    for model_name, values in raw_results.iterrows():
        metric_map = values.to_dict()
        training_time = metric_map.get("Time Taken")
        if training_time is not None and pd.notna(training_time):
            training_time_value = float(training_time)
        else:
            training_time_value = None

        rows.append(
            BenchmarkResultRow(
                model_name=str(model_name),
                task_type=task_type,
                raw_metrics=metric_map,
                training_time_seconds=training_time_value,
                run_timestamp=timestamp,
                benchmark_backend=benchmark_backend,
            )
        )

    return rows


def leaderboard_to_dataframe(rows: list[BenchmarkResultRow]) -> pd.DataFrame:
    """Convert normalized leaderboard rows to a flat dataframe."""

    records: list[dict[str, object]] = []
    for row in rows:
        record: dict[str, object] = {
            "Rank": row.rank,
            "Model": row.model_name,
            "Task": row.task_type.value.replace("_", " ").title(),
            "Score": row.primary_score,
            "Training Time (s)": row.training_time_seconds,
            "Backend": row.benchmark_backend.value.title(),
            "Run Time": row.run_timestamp.isoformat(),
            "Warnings": "; ".join(row.warnings),
        }
        for key, value in row.raw_metrics.items():
            record[key] = value
        records.append(record)
    return pd.DataFrame(records)


def build_benchmark_summary(
    *,
    dataset_name: str | None,
    dataset_fingerprint: str | None,
    config: BenchmarkConfig,
    task_type: BenchmarkTaskType,
    benchmark_backend: ExecutionBackend,
    workspace_mode: WorkspaceMode | None,
    ranking_metric: str,
    ranking_direction: BenchmarkSortDirection,
    ranked_rows: list[BenchmarkResultRow],
    source_row_count: int,
    source_column_count: int,
    benchmark_row_count: int,
    feature_column_count: int,
    train_row_count: int,
    test_row_count: int,
    sampled_row_count: int | None,
    stratified_split_applied: bool,
    benchmark_duration_seconds: float,
    warnings: list[str],
) -> BenchmarkSummary:
    """Build the roll-up benchmark summary from ranked rows and metadata."""

    best_row = next((row for row in ranked_rows if row.primary_score is not None), None)
    fastest_row = min(
        (row for row in ranked_rows if row.training_time_seconds is not None),
        key=lambda row: row.training_time_seconds,
        default=None,
    )

    return BenchmarkSummary(
        dataset_name=dataset_name,
        dataset_fingerprint=dataset_fingerprint,
        target_column=config.target_column,
        task_type=task_type,
        benchmark_backend=benchmark_backend,
        workspace_mode=workspace_mode,
        ranking_metric=ranking_metric,
        ranking_direction=ranking_direction,
        source_row_count=source_row_count,
        source_column_count=source_column_count,
        benchmark_row_count=benchmark_row_count,
        feature_column_count=feature_column_count,
        train_row_count=train_row_count,
        test_row_count=test_row_count,
        sampled_row_count=sampled_row_count,
        stratified_split_applied=stratified_split_applied,
        model_count=len(ranked_rows),
        best_model_name=best_row.model_name if best_row else None,
        best_score=best_row.primary_score if best_row else None,
        fastest_model_name=fastest_row.model_name if fastest_row else None,
        fastest_model_time_seconds=fastest_row.training_time_seconds if fastest_row else None,
        benchmark_duration_seconds=round(benchmark_duration_seconds, 4),
        warnings=warnings,
        split_config=config.split,
    )