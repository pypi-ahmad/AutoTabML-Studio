"""Task selection, target validation, and reliability heuristics for benchmarks."""

from __future__ import annotations

import pandas as pd

from app.modeling.benchmark.errors import BenchmarkTargetError, UnsupportedBenchmarkTaskError
from app.modeling.benchmark.schemas import BenchmarkSplitConfig, BenchmarkTaskType


def resolve_task_type(
    target: pd.Series,
    requested_task_type: BenchmarkTaskType,
) -> tuple[BenchmarkTaskType, list[str]]:
    """Resolve the effective task type and validate the target."""

    warnings: list[str] = []
    non_null_target = target.dropna()
    if non_null_target.empty:
        raise BenchmarkTargetError("Benchmark target is entirely null.")

    if requested_task_type == BenchmarkTaskType.AUTO:
        effective_task_type = infer_task_type(non_null_target)
        warnings.append(f"Task type auto-detected as {effective_task_type.value}.")
    else:
        effective_task_type = requested_task_type

    warnings.extend(validate_target(non_null_target, effective_task_type))
    return effective_task_type, warnings


def infer_task_type(target: pd.Series) -> BenchmarkTaskType:
    """Infer a benchmark task type from the target values."""

    n_unique = int(target.nunique(dropna=True))
    n_rows = len(target)

    if not pd.api.types.is_numeric_dtype(target):
        return BenchmarkTaskType.CLASSIFICATION

    if pd.api.types.is_bool_dtype(target):
        return BenchmarkTaskType.CLASSIFICATION

    if pd.api.types.is_integer_dtype(target) and n_unique <= min(20, max(2, int(n_rows ** 0.5))):
        return BenchmarkTaskType.CLASSIFICATION

    if n_unique <= min(10, max(2, int(n_rows * 0.02))):
        return BenchmarkTaskType.CLASSIFICATION

    return BenchmarkTaskType.REGRESSION


def validate_target(target: pd.Series, task_type: BenchmarkTaskType) -> list[str]:
    """Validate a target series for the requested benchmark task."""

    warnings: list[str] = []
    n_unique = int(target.nunique(dropna=True))

    if task_type == BenchmarkTaskType.CLASSIFICATION:
        if n_unique < 2:
            raise BenchmarkTargetError(
                "Classification target must contain at least two classes."
            )

        class_counts = target.value_counts(dropna=True)
        if not class_counts.empty and int(class_counts.min()) < 2:
            warnings.append(
                "Some classes have fewer than 2 rows; stratified split may not be possible."
            )

        if n_unique > min(len(target) // 2, 100):
            warnings.append(
                "Classification target has very high cardinality; baseline results may be unstable."
            )
        return warnings

    if task_type == BenchmarkTaskType.REGRESSION:
        numeric_target = pd.to_numeric(target, errors="coerce")
        if numeric_target.isna().any():
            raise BenchmarkTargetError(
                "Regression target must be numeric or cleanly coercible to numeric values."
            )
        if numeric_target.nunique(dropna=True) < 2 or float(numeric_target.std()) == 0.0:
            raise BenchmarkTargetError(
                "Regression target must not be constant."
            )
        if len(numeric_target) < 30:
            warnings.append(
                "Regression benchmark is running on a very small target sample; rankings may be noisy."
            )
        return warnings

    raise UnsupportedBenchmarkTaskError(
        f"Unsupported benchmark task type: {task_type.value}."
    )


def choose_stratify_target(
    target: pd.Series,
    task_type: BenchmarkTaskType,
    split_config: BenchmarkSplitConfig,
) -> tuple[pd.Series | None, bool, list[str]]:
    """Choose whether stratification should be applied for the split."""

    warnings: list[str] = []
    if task_type != BenchmarkTaskType.CLASSIFICATION:
        return None, False, warnings

    if split_config.stratify is False:
        return None, False, warnings

    class_counts = target.value_counts(dropna=True)
    if class_counts.empty or int(class_counts.min()) < 2:
        warnings.append(
            "Stratified split was skipped because at least one class had fewer than 2 rows."
        )
        return None, False, warnings

    estimated_test_rows = max(1, int(round(len(target) * split_config.test_size)))
    if estimated_test_rows < int(target.nunique(dropna=True)):
        warnings.append(
            "Stratified split was skipped because the test split would contain fewer rows than classes."
        )
        return None, False, warnings

    return target, True, warnings


def benchmark_reliability_warnings(
    frame: pd.DataFrame,
    target_column: str,
    task_type: BenchmarkTaskType,
) -> list[str]:
    """Emit lightweight warnings about benchmark reliability."""

    warnings: list[str] = []
    row_count = len(frame)
    feature_count = max(0, len(frame.columns) - 1)

    if row_count < 50:
        warnings.append("Benchmark is running on fewer than 50 rows; leaderboard stability may be limited.")

    if feature_count == 0:
        warnings.append("No feature columns remain after removing the target column.")

    if feature_count > row_count:
        warnings.append(
            "Feature count exceeds row count; baseline metrics may overstate generalization."
        )

    feature_frame = frame.drop(columns=[target_column], errors="ignore")
    missing_ratio = float(feature_frame.isna().mean().mean()) if not feature_frame.empty else 0.0
    if missing_ratio >= 0.2:
        warnings.append(
            "Features contain substantial missingness; LazyPredict comparisons may be sensitive to imputations."
        )

    if task_type == BenchmarkTaskType.CLASSIFICATION:
        class_share = frame[target_column].value_counts(normalize=True, dropna=True)
        if not class_share.empty and float(class_share.min()) <= 0.05:
            warnings.append(
                "Classification target is highly imbalanced; compare models with care."
            )

    return warnings


def collect_nested_object_columns(feature_frame: pd.DataFrame) -> list[str]:
    """Return object-like columns containing nested Python objects."""

    nested_columns: list[str] = []
    for column in feature_frame.select_dtypes(include=["object", "string"]).columns:
        sample = feature_frame[column].dropna().head(100)
        if any(_is_nested_value(value) for value in sample):
            nested_columns.append(str(column))
    return nested_columns


def _is_nested_value(value: object) -> bool:
    return isinstance(value, (dict, list, tuple, set))