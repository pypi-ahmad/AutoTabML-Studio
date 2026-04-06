"""MLflow tracking wrapper for benchmark runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.modeling.benchmark.schemas import BenchmarkArtifactBundle, BenchmarkResultBundle

logger = logging.getLogger(__name__)


def is_mlflow_available() -> bool:
    """Return True if mlflow is importable."""

    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def _get_mlflow_module() -> Any:
    import mlflow

    return mlflow


class MLflowBenchmarkTracker:
    """Lightweight MLflow tracking wrapper for benchmark runs."""

    def __init__(
        self,
        experiment_name: str,
        *,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._registry_uri = registry_uri

    def log_benchmark_run(self, bundle: BenchmarkResultBundle) -> tuple[str | None, list[str]]:
        """Log benchmark params, metrics, and artifacts to MLflow."""

        warnings: list[str] = []
        if not is_mlflow_available():
            warnings.append("MLflow tracking skipped because mlflow is not installed.")
            return None, warnings

        mlflow = _get_mlflow_module()

        try:
            if self._tracking_uri:
                mlflow.set_tracking_uri(self._tracking_uri)
            if self._registry_uri:
                mlflow.set_registry_uri(self._registry_uri)
            mlflow.set_experiment(self._experiment_name)
            with mlflow.start_run(run_name=_build_run_name(bundle)) as run:
                mlflow.log_params(_build_params(bundle))
                mlflow.log_metrics(_build_metrics(bundle))
                _log_artifacts(mlflow, bundle.artifacts)
                return run.info.run_id, warnings
        except Exception as exc:
            logger.warning("MLflow benchmark tracking failed: %s", exc)
            warnings.append(f"MLflow tracking failed: {exc}")
            return None, warnings


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