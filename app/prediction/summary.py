"""Summary helpers for prediction workflows."""

from __future__ import annotations

from pathlib import Path

from app.prediction.schemas import (
    LoadedModel,
    PredictionHistoryEntry,
    PredictionMode,
    PredictionStatus,
    PredictionSummary,
    SchemaValidationMode,
)
from app.path_utils import safe_artifact_stem


def build_prediction_summary(
    *,
    mode: PredictionMode,
    loaded_model: LoadedModel,
    input_source: str,
    input_row_count: int,
    rows_scored: int,
    rows_failed: int,
    prediction_column: str,
    prediction_score_column: str | None,
    validation_mode: SchemaValidationMode,
    warnings: list[str],
    status: PredictionStatus = PredictionStatus.SUCCESS,
    output_artifact_path: Path | None = None,
) -> PredictionSummary:
    """Build a normalized prediction summary record."""

    return PredictionSummary(
        status=status,
        mode=mode,
        source_type=loaded_model.source_type,
        task_type=loaded_model.task_type,
        model_identifier=loaded_model.model_identifier,
        input_source=input_source,
        input_row_count=input_row_count,
        rows_scored=rows_scored,
        rows_failed=rows_failed,
        output_artifact_path=output_artifact_path,
        prediction_column=prediction_column,
        prediction_score_column=prediction_score_column,
        validation_mode=validation_mode,
        warnings=list(warnings),
    )


def build_history_entry(
    summary: PredictionSummary,
    *,
    summary_json_path: Path | None = None,
    metadata_json_path: Path | None = None,
) -> PredictionHistoryEntry:
    """Build a lightweight history entry from a prediction summary."""

    return PredictionHistoryEntry(
        job_id=prediction_job_id(summary),
        timestamp=summary.run_timestamp,
        status=summary.status,
        mode=summary.mode,
        model_source=summary.source_type,
        model_identifier=summary.model_identifier,
        task_type=summary.task_type,
        input_source=summary.input_source,
        row_count=summary.input_row_count,
        output_artifact_path=summary.output_artifact_path,
        summary_json_path=summary_json_path,
        metadata_json_path=metadata_json_path,
    )


def prediction_job_id(summary: PredictionSummary) -> str:
    """Return a stable-ish job id derived from summary fields."""

    timestamp = summary.run_timestamp.strftime("%Y%m%dT%H%M%S")
    stem = safe_artifact_stem(summary.model_identifier or "model", default="model")
    return f"{summary.mode.value}_{stem}_{timestamp}"


def prediction_summary_line(summary: PredictionSummary) -> str:
    """Return a compact one-line summary for CLI output."""

    return (
        f"[{summary.status.value}] {summary.mode.value} | "
        f"model={summary.model_identifier} | rows={summary.rows_scored}/{summary.input_row_count}"
    )
