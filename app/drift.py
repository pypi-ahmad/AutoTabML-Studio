"""Aggregate, row-free training baselines and prediction-data drift checks."""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class DriftLevel(str, Enum):
    STABLE = "stable"
    WARNING = "warning"
    HIGH = "high"


class FeatureBaseline(BaseModel):
    kind: str
    missing_rate: float
    bins: list[float] = Field(default_factory=list)
    proportions: list[float] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


class DriftBaseline(BaseModel):
    row_count: int
    features: dict[str, FeatureBaseline]


class FeatureDrift(BaseModel):
    feature: str
    psi: float | None = None
    missing_rate_delta: float | None = None
    level: DriftLevel
    message: str


class DriftReport(BaseModel):
    level: DriftLevel
    row_count: int
    features: list[FeatureDrift]
    missing_columns: list[str] = Field(default_factory=list)
    unexpected_columns: list[str] = Field(default_factory=list)


def build_drift_baseline(dataframe: pd.DataFrame, *, max_rows: int = 50_000) -> DriftBaseline:
    frame = dataframe.sample(max_rows, random_state=42) if len(dataframe) > max_rows else dataframe
    features: dict[str, FeatureBaseline] = {}
    for name in frame.columns:
        series = frame[name]
        missing = float(series.isna().mean())
        if pd.api.types.is_numeric_dtype(series):
            clean = pd.to_numeric(series, errors="coerce").dropna()
            bins = sorted({float(v) for v in clean.quantile(np.linspace(0, 1, 11)).tolist()})
            if len(bins) < 2:
                bins = [float(clean.min()) - 1.0, float(clean.max()) + 1.0] if len(clean) else [0.0, 1.0]
            bins[0], bins[-1] = float("-inf"), float("inf")
            features[str(name)] = FeatureBaseline(
                kind="numeric",
                missing_rate=missing,
                bins=bins,
                proportions=_numeric_proportions(series, bins),
            )
        else:
            normalized = series.fillna("<MISSING>").astype(str)
            categories = normalized.value_counts().head(50).index.tolist()
            features[str(name)] = FeatureBaseline(
                kind="categorical",
                missing_rate=missing,
                categories=categories,
                proportions=_categorical_proportions(normalized, categories),
            )
    return DriftBaseline(row_count=len(dataframe), features=features)


def compare_drift(baseline: DriftBaseline, dataframe: pd.DataFrame) -> DriftReport:
    missing = sorted(set(baseline.features) - set(dataframe.columns))
    unexpected = sorted(set(dataframe.columns) - set(baseline.features))
    results: list[FeatureDrift] = []
    for name, reference in baseline.features.items():
        if name not in dataframe:
            results.append(FeatureDrift(feature=name, level=DriftLevel.HIGH, message="Feature is missing."))
            continue
        series = dataframe[name]
        observed = (
            _numeric_proportions(series, reference.bins)
            if reference.kind == "numeric"
            else _categorical_proportions(series.fillna("<MISSING>").astype(str), reference.categories)
        )
        psi = _psi(reference.proportions, observed)
        missing_delta = abs(float(series.isna().mean()) - reference.missing_rate)
        level = _level(psi)
        if missing_delta >= 0.10 and level == DriftLevel.STABLE:
            level = DriftLevel.WARNING
        results.append(
            FeatureDrift(
                feature=name,
                psi=psi,
                missing_rate_delta=missing_delta,
                level=level,
                message=f"PSI {psi:.3f}; missing-rate change {missing_delta:.1%}.",
            )
        )
    overall = (
        DriftLevel.HIGH
        if missing or any(result.level == DriftLevel.HIGH for result in results)
        else DriftLevel.WARNING
        if unexpected or any(result.level == DriftLevel.WARNING for result in results)
        else DriftLevel.STABLE
    )
    return DriftReport(
        level=overall,
        row_count=len(dataframe),
        features=results,
        missing_columns=missing,
        unexpected_columns=unexpected,
    )


def _numeric_proportions(series: pd.Series, bins: list[float]) -> list[float]:
    values = pd.to_numeric(series, errors="coerce")
    counts = pd.cut(values, bins=bins, include_lowest=True).value_counts(sort=False)
    total = max(int(counts.sum()), 1)
    return [float(value / total) for value in counts]


def _categorical_proportions(series: pd.Series, categories: list[str]) -> list[float]:
    values = series.where(series.isin(categories), "<OTHER>")
    counts = values.value_counts()
    total = max(len(values), 1)
    return [float(counts.get(name, 0) / total) for name in [*categories, "<OTHER>"]]


def _psi(expected: list[float], actual: list[float]) -> float:
    size = max(len(expected), len(actual))
    left = np.pad(np.asarray(expected, dtype=float), (0, size - len(expected)), constant_values=0)
    right = np.pad(np.asarray(actual, dtype=float), (0, size - len(actual)), constant_values=0)
    left, right = np.clip(left, 1e-6, None), np.clip(right, 1e-6, None)
    return float(np.sum((right - left) * np.log(right / left)))


def _level(psi: float) -> DriftLevel:
    if psi >= 0.25:
        return DriftLevel.HIGH
    if psi >= 0.10:
        return DriftLevel.WARNING
    return DriftLevel.STABLE


__all__ = ["DriftBaseline", "DriftLevel", "DriftReport", "build_drift_baseline", "compare_drift"]
