"""Engine-neutral holdout evaluation for Auto Run."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class EvaluationReport(BaseModel):
    task_type: str
    row_count: int
    metrics: dict[str, float]
    warnings: list[str] = Field(default_factory=list)
    calibration: dict[str, float] = Field(default_factory=dict)


def evaluate_model(model: Any, features: pd.DataFrame, target: pd.Series, *, task_type: str) -> EvaluationReport:
    """Evaluate a fitted sklearn-compatible model on untouched data."""

    from sklearn import metrics

    predicted = model.predict(features)
    warnings: list[str] = []
    values: dict[str, float]
    calibration: dict[str, float] = {}
    if task_type == "classification":
        values = {
            "accuracy": float(metrics.accuracy_score(target, predicted)),
            "balanced_accuracy": float(metrics.balanced_accuracy_score(target, predicted)),
            "f1_macro": float(metrics.f1_score(target, predicted, average="macro", zero_division=0)),
        }
        counts = target.value_counts(normalize=True)
        if len(counts) > 1 and float(counts.min()) < 0.20:
            warnings.append("Target is imbalanced; prefer balanced accuracy and macro F1.")
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(features))
            if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                positive = probabilities[:, 1]
                try:
                    values["roc_auc"] = float(metrics.roc_auc_score(target, positive))
                    calibration["brier_score"] = float(metrics.brier_score_loss(target, positive))
                except ValueError:
                    warnings.append("Probability metrics were unavailable for this holdout.")
    else:
        values = {
            "mae": float(metrics.mean_absolute_error(target, predicted)),
            "rmse": float(metrics.root_mean_squared_error(target, predicted)),
            "r2": float(metrics.r2_score(target, predicted)),
        }
    return EvaluationReport(
        task_type=task_type,
        row_count=len(features),
        metrics=values,
        warnings=warnings,
        calibration=calibration,
    )


__all__ = ["EvaluationReport", "evaluate_model"]
