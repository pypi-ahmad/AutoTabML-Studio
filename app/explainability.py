"""Bounded model explanations with dependency-free fallbacks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    feature: str
    value: float


class ModelExplanation(BaseModel):
    method: str
    contributions: list[FeatureContribution]
    warnings: list[str] = Field(default_factory=list)


def explain_global(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    max_rows: int = 500,
) -> ModelExplanation:
    """Return global feature importance using native or permutation importance."""

    names = list(features.columns)
    shap_values = _shap_importance(model, features.head(min(max_rows, 100)))
    if shap_values is not None and len(shap_values) == len(names):
        values = shap_values
        method = "shap"
    else:
        native = getattr(model, "feature_importances_", None)
    if shap_values is None and native is not None and len(native) == len(names):
        values = np.abs(np.asarray(native, dtype=float))
        method = "native_feature_importance"
    elif shap_values is None:
        coefficients = getattr(model, "coef_", None)
        if coefficients is not None and np.asarray(coefficients).shape[-1] == len(names):
            values = np.abs(np.asarray(coefficients, dtype=float)).reshape(-1, len(names)).mean(axis=0)
            method = "absolute_coefficient"
        else:
            from sklearn.inspection import permutation_importance

            sample = features.head(max_rows)
            result = permutation_importance(model, sample, target.loc[sample.index], n_repeats=3, random_state=42)
            values = np.abs(result.importances_mean)
            method = "permutation_importance"
    ordered = sorted(zip(names, values, strict=True), key=lambda item: item[1], reverse=True)
    return ModelExplanation(
        method=method,
        contributions=[FeatureContribution(feature=name, value=float(value)) for name, value in ordered],
    )


def _shap_importance(model: Any, sample: pd.DataFrame) -> np.ndarray | None:
    try:
        import shap

        values = np.asarray(shap.Explainer(model, sample)(sample).values, dtype=float)
        if values.ndim == 2:
            return np.abs(values).mean(axis=0)
        if values.ndim == 3:
            return np.abs(values).mean(axis=(0, 2))
    except (ImportError, TypeError, ValueError, AttributeError):
        return None
    return None


def explain_prediction(model: Any, row: pd.DataFrame, reference: pd.DataFrame) -> ModelExplanation:
    """Approximate one prediction by replacing each feature with a reference value."""

    if len(row) != 1:
        raise ValueError("A local explanation requires exactly one row.")
    base = reference.median(numeric_only=True).to_dict()
    for column in reference.columns:
        if column not in base:
            modes = reference[column].mode(dropna=True)
            base[column] = modes.iloc[0] if not modes.empty else None
    baseline = pd.DataFrame([base], columns=row.columns)
    original_score = _score(model, row)
    values: list[FeatureContribution] = []
    for column in row.columns:
        changed = row.copy()
        if pd.api.types.is_numeric_dtype(reference[column]):
            changed[column] = pd.to_numeric(changed[column], errors="coerce").astype(float)
        else:
            changed[column] = changed[column].astype(object)
        changed.loc[changed.index[0], column] = baseline.loc[0, column]
        values.append(FeatureContribution(feature=str(column), value=original_score - _score(model, changed)))
    values.sort(key=lambda item: abs(item.value), reverse=True)
    return ModelExplanation(
        method="single_feature_perturbation",
        contributions=values,
        warnings=["Approximate association, not a causal explanation."],
    )


def _score(model: Any, frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(frame))
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return float(probabilities[0, -1])
    return float(np.asarray(model.predict(frame))[0])


__all__ = ["ModelExplanation", "explain_global", "explain_prediction"]
