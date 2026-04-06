"""Single-row prediction helpers."""

from __future__ import annotations

import pandas as pd

from app.prediction.schemas import (
    LoadedModel,
    PredictionArtifactBundle,
    PredictionHistoryEntry,
    PredictionResult,
    PredictionSummary,
    PredictionTaskType,
    PredictionValidationResult,
)


def build_single_prediction_result(
    *,
    loaded_model: LoadedModel,
    scored_dataframe: pd.DataFrame,
    validation: PredictionValidationResult,
    summary: PredictionSummary,
    artifacts: PredictionArtifactBundle | None,
    history_entry: PredictionHistoryEntry | None,
    warnings: list[str],
    prediction_column_name: str,
    prediction_score_column_name: str,
) -> PredictionResult:
    """Build a normalized single-row prediction result."""

    row = scored_dataframe.iloc[0]
    raw_prediction = row[prediction_column_name]
    predicted_label = raw_prediction if loaded_model.task_type == PredictionTaskType.CLASSIFICATION else None
    predicted_value = raw_prediction if loaded_model.task_type != PredictionTaskType.CLASSIFICATION else None
    predicted_score = None
    if prediction_score_column_name in scored_dataframe.columns:
        score_value = row[prediction_score_column_name]
        predicted_score = float(score_value) if score_value is not None else None

    return PredictionResult(
        loaded_model=loaded_model,
        validation=validation,
        summary=summary,
        artifacts=artifacts,
        history_entry=history_entry,
        warnings=list(warnings),
        row_index=row.name,
        predicted_label=predicted_label,
        predicted_value=predicted_value,
        predicted_score=predicted_score,
        scored_row=row.to_dict(),
    )