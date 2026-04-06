"""Validation helpers for prediction-time inputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.prediction.errors import PredictionValidationError
from app.prediction.schemas import (
    LoadedModel,
    PredictionValidationIssue,
    PredictionValidationResult,
    PredictionValidationSeverity,
    SchemaValidationMode,
)


def normalize_single_row_input(row_data: dict[str, Any]) -> pd.DataFrame:
    """Normalize one row of dict-like input into a one-row dataframe."""

    if not isinstance(row_data, dict):
        raise PredictionValidationError("Single-row prediction requires a JSON object / dict payload.")
    if not row_data:
        raise PredictionValidationError("Single-row prediction payload must not be empty.")
    return pd.DataFrame([row_data])


def validate_single_row_shape(dataframe: pd.DataFrame) -> None:
    """Ensure the dataframe represents exactly one row."""

    if dataframe.empty:
        raise PredictionValidationError("Prediction input is empty.")
    if len(dataframe.index) != 1:
        raise PredictionValidationError("Single-row prediction requires exactly one row of input.")


def validate_prediction_dataframe(
    dataframe: pd.DataFrame,
    loaded_model: LoadedModel,
    *,
    validation_mode: SchemaValidationMode,
) -> tuple[pd.DataFrame, PredictionValidationResult]:
    """Validate and normalize batch/single-row prediction inputs."""

    if dataframe.empty:
        raise PredictionValidationError("Prediction input is empty.")

    normalized = dataframe.copy()
    issues: list[PredictionValidationIssue] = []
    missing_columns: list[str] = []
    unexpected_columns: list[str] = []
    metadata_available = bool(loaded_model.feature_columns)

    if loaded_model.feature_columns:
        expected_columns = list(dict.fromkeys(loaded_model.feature_columns))
        missing_columns = [column for column in expected_columns if column not in normalized.columns]
        unexpected_columns = [column for column in normalized.columns if column not in expected_columns]

        if missing_columns:
            issues.append(
                PredictionValidationIssue(
                    severity=PredictionValidationSeverity.ERROR,
                    message="Missing required columns for prediction: " + ", ".join(missing_columns),
                )
            )

        if unexpected_columns:
            severity = (
                PredictionValidationSeverity.ERROR
                if validation_mode == SchemaValidationMode.STRICT
                else PredictionValidationSeverity.WARNING
            )
            issues.append(
                PredictionValidationIssue(
                    severity=severity,
                    message="Unexpected input columns: " + ", ".join(unexpected_columns),
                )
            )
            if validation_mode == SchemaValidationMode.WARN:
                normalized = normalized.drop(columns=unexpected_columns, errors="ignore")

        if not missing_columns and set(expected_columns).issubset(set(normalized.columns)):
            normalized = normalized.reindex(columns=expected_columns)

        feature_dtypes = loaded_model.metadata.get("feature_dtypes", {})
        if isinstance(feature_dtypes, dict):
            for column, expected_dtype in feature_dtypes.items():
                if column not in normalized.columns:
                    continue
                actual_dtype = str(normalized[column].dtype)
                if expected_dtype and actual_dtype != expected_dtype:
                    issues.append(
                        PredictionValidationIssue(
                            severity=PredictionValidationSeverity.WARNING,
                            field_name=column,
                            message=(
                                f"Column '{column}' dtype is '{actual_dtype}', expected '{expected_dtype}'. "
                                "Prediction will be attempted because dtype compatibility cannot be guaranteed."
                            ),
                        )
                    )
    else:
        issues.append(
            PredictionValidationIssue(
                severity=PredictionValidationSeverity.WARNING,
                message=(
                    "Saved model metadata does not include feature columns, so feature-level schema "
                    "compatibility could not be validated. Prediction will be attempted using the "
                    "columns provided."
                ),
            )
        )

    result = PredictionValidationResult(
        can_score=not any(issue.severity == PredictionValidationSeverity.ERROR for issue in issues),
        metadata_available=metadata_available,
        missing_columns=missing_columns,
        unexpected_columns=unexpected_columns,
        normalized_columns=list(normalized.columns),
        issues=issues,
    )
    return normalized, result


def ensure_validation_can_score(result: PredictionValidationResult) -> None:
    """Raise a clean exception when validation produced blocking errors."""

    if result.can_score:
        return
    raise PredictionValidationError("; ".join(result.errors))