"""MLflow tracking wrapper for benchmark runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.modeling.base import (
    BaseTracker,
    get_mlflow_module as _base_get_mlflow_module,
    is_mlflow_available as _base_is_mlflow_available,
    mlflow_exception_types as _base_mlflow_exception_types,
)
from app.modeling.benchmark.schemas import BenchmarkArtifactBundle, BenchmarkResultBundle

logger = logging.getLogger(__name__)


def is_mlflow_available() -> bool:
    """Return True if mlflow is importable."""

    return _base_is_mlflow_available()


def _get_mlflow_module() -> Any:
    return _base_get_mlflow_module()


def _mlflow_exception_types(mlflow_module: Any) -> tuple[type[BaseException], ...]:
    return _base_mlflow_exception_types(mlflow_module)


class MLflowBenchmarkTracker(BaseTracker[BenchmarkResultBundle]):
    """Lightweight MLflow tracking wrapper for benchmark runs."""

    def log_benchmark_run(self, bundle: BenchmarkResultBundle) -> tuple[str | None, list[str]]:
        """Backward-compatible benchmark tracker entrypoint."""

        return self.log_bundle(bundle)

    def _is_mlflow_available(self) -> bool:
        return is_mlflow_available()

    def _get_mlflow_module(self) -> Any:
        return _get_mlflow_module()

    def _operation_name(self) -> str:
        return "benchmark.mlflow_tracking"

    def _build_run_name(self, bundle: BenchmarkResultBundle) -> str:
        return _build_run_name(bundle)

    def _build_params(self, bundle: BenchmarkResultBundle) -> dict[str, Any]:
        return _build_params(bundle)

    def _build_metrics(self, bundle: BenchmarkResultBundle) -> dict[str, float]:
        return _build_metrics(bundle)

    def _artifact_paths(self, bundle: BenchmarkResultBundle) -> list[Path | None]:
        artifacts = bundle.artifacts
        if artifacts is None:
            return []
        return [
            artifacts.raw_results_csv_path,
            artifacts.leaderboard_csv_path,
            artifacts.leaderboard_json_path,
            artifacts.summary_json_path,
            artifacts.markdown_summary_path,
            artifacts.score_chart_path,
            artifacts.training_time_chart_path,
        ]


def _build_run_name(bundle: BenchmarkResultBundle) -> str:
    dataset_name = bundle.dataset_name or "dataset"
    return f"benchmark-{bundle.task_type.value}-{dataset_name}"


def _build_params(bundle: BenchmarkResultBundle) -> dict[str, Any]:
    summary = bundle.summary
    split = bundle.config.split
    return {
        "task_type": bundle.task_type.value,
        "target_column": bundle.config.target_column,
        "dataset_fingerprint": bundle.dataset_fingerprint or "",
        "test_size": split.test_size,
        "random_state": split.random_state,
        "stratify_requested": split.stratify if split.stratify is not None else "auto",
        "stratify_applied": summary.stratified_split_applied,
        "ranking_metric": summary.ranking_metric,
        "execution_backend": bundle.benchmark_backend.value,
        "workspace_mode": bundle.workspace_mode.value if bundle.workspace_mode else "",
        "source_row_count": summary.source_row_count,
        "source_column_count": summary.source_column_count,
        "benchmark_row_count": summary.benchmark_row_count,
        "feature_column_count": summary.feature_column_count,
        "sampled_row_count": summary.sampled_row_count or 0,
    }


def _build_metrics(bundle: BenchmarkResultBundle) -> dict[str, float]:
    summary = bundle.summary
    metrics: dict[str, float] = {
        "model_count": float(summary.model_count),
        "benchmark_duration_seconds": float(summary.benchmark_duration_seconds),
    }
    if summary.best_score is not None:
        metrics["best_score"] = float(summary.best_score)
    if summary.fastest_model_time_seconds is not None:
        metrics["fastest_model_time_seconds"] = float(summary.fastest_model_time_seconds)
    return metrics


def _log_artifacts(mlflow: Any, artifacts: BenchmarkArtifactBundle | None) -> None:
    if artifacts is None:
        return

    for path in [
        artifacts.raw_results_csv_path,
        artifacts.leaderboard_csv_path,
        artifacts.leaderboard_json_path,
        artifacts.summary_json_path,
        artifacts.markdown_summary_path,
        artifacts.score_chart_path,
        artifacts.training_time_chart_path,
    ]:
        if path is not None and Path(path).exists():
            mlflow.log_artifact(str(path))