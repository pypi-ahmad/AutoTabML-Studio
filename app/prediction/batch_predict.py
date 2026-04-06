"""Batch prediction helpers."""

from __future__ import annotations

import pandas as pd

from app.ingestion import load_dataset
from app.prediction.errors import PredictionValidationError
from app.prediction.schemas import (
    BatchPredictionRequest,
    BatchPredictionResult,
    LoadedModel,
    PredictionArtifactBundle,
    PredictionHistoryEntry,
    PredictionInputSourceType,
    PredictionSummary,
    PredictionValidationResult,
)


def resolve_batch_dataframe(request: BatchPredictionRequest) -> tuple[pd.DataFrame, str]:
    """Resolve one batch request into a normalized dataframe and source label."""

    if request.dataframe is not None:
        label = request.dataset_name or request.input_source_label or "dataframe"
        return request.dataframe.copy(), label

    if request.input_spec is not None:
        loaded = load_dataset(request.input_spec)
        label = request.dataset_name or loaded.metadata.display_name or loaded.metadata.source_locator
        return loaded.dataframe.copy(), label

    raise PredictionValidationError("Batch prediction requires a dataframe or an ingestion input spec.")


def infer_input_source_type(request: BatchPredictionRequest) -> PredictionInputSourceType:
    """Return the input source type for a batch request."""

    if request.input_spec is not None and request.input_spec.path is not None:
        return PredictionInputSourceType.FILE
    return request.input_source_type


def build_batch_prediction_result(
    *,
    loaded_model: LoadedModel,
    scored_dataframe: pd.DataFrame,
    validation: PredictionValidationResult,
    summary: PredictionSummary,
    artifacts: PredictionArtifactBundle | None,
    history_entry: PredictionHistoryEntry | None,
    warnings: list[str],
) -> BatchPredictionResult:
    """Build a normalized batch prediction result."""

    return BatchPredictionResult(
        loaded_model=loaded_model,
        validation=validation,
        summary=summary,
        artifacts=artifacts,
        history_entry=history_entry,
        warnings=list(warnings),
        scored_dataframe=scored_dataframe,
    )