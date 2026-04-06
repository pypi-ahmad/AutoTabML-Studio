"""Side-by-side run comparison service."""

from __future__ import annotations

from app.tracking.schemas import (
    ComparisonBundle,
    ConfigDifference,
    MetricDelta,
    RunHistoryItem,
)


class ComparisonService:
    """Produce structured comparisons between two runs."""

    def compare(self, left: RunHistoryItem, right: RunHistoryItem) -> ComparisonBundle:
        """Compare two runs and return a structured bundle with deltas and warnings."""

        warnings = _check_comparability(left, right)
        metric_deltas = _compute_metric_deltas(left, right)
        config_diffs = _compute_config_differences(left, right)
        comparable = len(warnings) == 0

        return ComparisonBundle(
            left=left,
            right=right,
            metric_deltas=metric_deltas,
            config_differences=config_diffs,
            comparable=comparable,
            warnings=warnings,
        )


_LOWER_IS_BETTER_HINTS = ("rmse", "mae", "mse", "rmsle", "mape", "loss", "error")


def _check_comparability(left: RunHistoryItem, right: RunHistoryItem) -> list[str]:
    warnings: list[str] = []

    if left.dataset_fingerprint and right.dataset_fingerprint:
        if left.dataset_fingerprint != right.dataset_fingerprint:
            warnings.append("Runs used different datasets.")
    elif left.dataset_name and right.dataset_name:
        if left.dataset_name != right.dataset_name:
            warnings.append(
                f"Runs reference different dataset names: '{left.dataset_name}' vs '{right.dataset_name}'."
            )
        warnings.append(
            "Dataset fingerprints are missing, so dataset equality could not be fully verified."
        )
    else:
        warnings.append(
            "Dataset identity could not be fully verified because dataset fingerprints are missing."
        )

    if left.target_column and right.target_column:
        if left.target_column != right.target_column:
            warnings.append(
                f"Runs targeted different columns: '{left.target_column}' vs '{right.target_column}'."
            )
    else:
        warnings.append(
            "Target-column metadata is missing on one or both runs, so label alignment could not be fully verified."
        )

    if left.task_type and right.task_type:
        if left.task_type != right.task_type:
            warnings.append(
                f"Runs used different task types: '{left.task_type}' vs '{right.task_type}'."
            )
    else:
        warnings.append(
            "Task-type metadata is missing on one or both runs, so task comparability could not be fully verified."
        )

    if left.primary_metric_name and right.primary_metric_name:
        if left.primary_metric_name != right.primary_metric_name:
            warnings.append(
                f"Runs optimized different metrics: "
                f"'{left.primary_metric_name}' vs '{right.primary_metric_name}'."
            )
    else:
        warnings.append(
            "Primary metric metadata is missing on one or both runs, so metric comparability could not be fully verified."
        )

    if left.run_type != right.run_type:
        warnings.append(
            f"Runs are of different types: '{left.run_type.value}' vs '{right.run_type.value}'."
        )
    elif left.run_type.value == "unknown":
        warnings.append("Run type could not be fully inferred for one or both runs.")

    return warnings


def _compute_metric_deltas(left: RunHistoryItem, right: RunHistoryItem) -> list[MetricDelta]:
    all_metric_names = sorted(set(left.metrics) | set(right.metrics))
    deltas: list[MetricDelta] = []

    for name in all_metric_names:
        left_val = left.metrics.get(name)
        right_val = right.metrics.get(name)
        delta = None
        better = None

        if left_val is not None and right_val is not None:
            delta = round(right_val - left_val, 6)
            lower_better = any(hint in name.lower() for hint in _LOWER_IS_BETTER_HINTS)
            if delta == 0.0:
                better = "tie"
            elif lower_better:
                better = "right" if delta < 0 else "left"
            else:
                better = "right" if delta > 0 else "left"

        deltas.append(
            MetricDelta(
                name=name,
                left_value=left_val,
                right_value=right_val,
                delta=delta,
                better_side=better,
            )
        )
    return deltas


_CONFIG_PARAMS = [
    "task_type",
    "target_column",
    "dataset_fingerprint",
    "execution_backend",
    "workspace_mode",
    "test_size",
    "train_size",
    "random_state",
    "ranking_metric",
    "compare_optimize_metric",
    "tune_optimize_metric",
    "setup_session_id",
    "setup_train_size",
    "setup_fold",
    "setup_fold_strategy",
    "setup_preprocess",
]


def _compute_config_differences(
    left: RunHistoryItem,
    right: RunHistoryItem,
) -> list[ConfigDifference]:
    all_keys = sorted(set(_CONFIG_PARAMS) & (set(left.params) | set(right.params)))
    diffs: list[ConfigDifference] = []

    for key in all_keys:
        left_val = left.params.get(key)
        right_val = right.params.get(key)
        if left_val != right_val:
            diffs.append(
                ConfigDifference(
                    key=key,
                    left_value=left_val,
                    right_value=right_val,
                    category="setup" if key.startswith("setup_") else "general",
                )
            )
    return diffs
