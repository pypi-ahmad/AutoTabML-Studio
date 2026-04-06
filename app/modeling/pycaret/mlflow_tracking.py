"""Explicit MLflow tracking for PyCaret experiment runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.modeling.pycaret.schemas import ExperimentArtifactBundle, ExperimentResultBundle

logger = logging.getLogger(__name__)


def is_mlflow_available() -> bool:
    """Return True when mlflow is importable."""

    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def _get_mlflow_module() -> Any:
    import mlflow

    return mlflow


class MLflowExperimentTracker:
    """Lightweight explicit MLflow tracker for experiment runs."""

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

    def log_experiment_bundle(
        self,
        bundle: ExperimentResultBundle,
        *,
        existing_run_id: str | None = None,
    ) -> tuple[str | None, list[str]]:
        """Log experiment params, metrics, and artifacts to MLflow."""

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
            with mlflow.start_run(
                run_name=_build_run_name(bundle),
                run_id=existing_run_id,
            ) as run:
                mlflow.log_params(_build_params(bundle))
                mlflow.log_metrics(_build_metrics(bundle))
                _log_artifacts(mlflow, bundle.artifacts)
                return run.info.run_id, warnings
        except Exception as exc:  # pragma: no cover - exercised through tests
            logger.warning("MLflow experiment tracking failed: %s", exc)
            warnings.append(f"MLflow tracking failed: {exc}")
            return existing_run_id, warnings


def _build_run_name(bundle: ExperimentResultBundle) -> str:
    dataset_name = bundle.dataset_name or "dataset"
    return f"experiment-{bundle.task_type.value}-{dataset_name}"


def _build_params(bundle: ExperimentResultBundle) -> dict[str, Any]:
    summary = bundle.summary
    params: dict[str, Any] = {
        "task_type": bundle.task_type.value,
        "target_column": bundle.config.target_column,
        "dataset_fingerprint": bundle.dataset_fingerprint or "",
        "workspace_mode": bundle.workspace_mode.value if bundle.workspace_mode else "",
        "execution_backend": bundle.execution_backend.value,
        "mlflow_tracking_mode": bundle.config.mlflow_tracking_mode.value,
        "compare_optimize_metric": summary.compare_optimize_metric or "",
        "tune_optimize_metric": summary.tune_optimize_metric or bundle.config.tune.optimize or "",
        "selected_model_id": summary.selected_model_id or "",
        "selected_model_name": summary.selected_model_name or "",
        "best_baseline_model_name": summary.best_baseline_model_name or "",
        "tuned_model_name": summary.tuned_model_name or "",
    }

    for key, value in summary.setup_config.actual_setup_kwargs.items():
        params[f"setup_{key}"] = _stringify_param(value)

    return params


def _build_metrics(bundle: ExperimentResultBundle) -> dict[str, float]:
    summary = bundle.summary
    metrics: dict[str, float] = {
        "experiment_duration_seconds": float(summary.experiment_duration_seconds),
        "compare_model_count": float(len(bundle.compare_leaderboard)),
    }
    if summary.best_baseline_score is not None:
        metrics["best_baseline_score"] = float(summary.best_baseline_score)
    if summary.tuned_score is not None:
        metrics["tuned_score"] = float(summary.tuned_score)
    return metrics


def _stringify_param(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def _log_artifacts(mlflow: Any, artifacts: ExperimentArtifactBundle | None) -> None:
    if artifacts is None:
        return

    paths = [
        artifacts.setup_json_path,
        artifacts.metrics_csv_path,
        artifacts.metrics_json_path,
        artifacts.compare_csv_path,
        artifacts.compare_json_path,
        artifacts.tune_json_path,
        artifacts.summary_json_path,
        artifacts.markdown_summary_path,
        artifacts.saved_model_metadata_path,
        artifacts.experiment_snapshot_metadata_path,
    ]
    paths.extend(plot.path for plot in artifacts.plot_artifacts)

    for path in paths:
        if path is not None and Path(path).exists():
            mlflow.log_artifact(str(path))