"""Explicit MLflow tracking for FLAML AutoML runs."""

from __future__ import annotations

import logging
from typing import Any

from app.modeling.flaml.schemas import FlamlResultBundle

logger = logging.getLogger(__name__)


def is_mlflow_available() -> bool:
    """Return True when mlflow is importable."""

    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


class MLflowFlamlTracker:
    """Lightweight explicit MLflow tracker for FLAML runs."""

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

    def log_flaml_bundle(
        self,
        bundle: FlamlResultBundle,
        *,
        existing_run_id: str | None = None,
    ) -> tuple[str | None, list[str]]:
        """Log FLAML params, metrics, and artifacts to MLflow."""

        warnings: list[str] = []
        if not is_mlflow_available():
            warnings.append("MLflow tracking skipped because mlflow is not installed.")
            return None, warnings

        import mlflow

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
                _log_artifacts(mlflow, bundle)
                return run.info.run_id, warnings
        except Exception as exc:
            logger.warning("MLflow FLAML tracking failed: %s", exc)
            warnings.append(f"MLflow tracking failed: {exc}")
            return existing_run_id, warnings


def _build_run_name(bundle: FlamlResultBundle) -> str:
    dataset_name = bundle.dataset_name or "dataset"
    task = bundle.task_type.value
    return f"flaml-{dataset_name}-{task}"


def _build_params(bundle: FlamlResultBundle) -> dict[str, Any]:
    params: dict[str, Any] = {
        "framework": "flaml",
        "task_type": bundle.task_type.value,
        "target_column": bundle.config.target_column,
        "execution_backend": bundle.execution_backend.value,
    }
    if bundle.search_result:
        params["best_estimator"] = bundle.search_result.best_estimator or ""
        params["metric"] = bundle.search_result.metric or ""
    params["time_budget"] = bundle.config.search.time_budget
    params["n_splits"] = bundle.config.search.n_splits
    return params


def _build_metrics(bundle: FlamlResultBundle) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if bundle.search_result:
        if bundle.search_result.best_loss is not None:
            metrics["best_loss"] = bundle.search_result.best_loss
        if bundle.search_result.best_config_train_time is not None:
            metrics["best_config_train_time"] = bundle.search_result.best_config_train_time
        if bundle.search_result.time_to_find_best is not None:
            metrics["time_to_find_best"] = bundle.search_result.time_to_find_best
    metrics["search_duration_seconds"] = bundle.summary.search_duration_seconds
    return metrics


def _log_artifacts(mlflow_module: Any, bundle: FlamlResultBundle) -> None:
    if bundle.artifacts is None:
        return
    for path_field in (
        "search_result_json_path",
        "leaderboard_csv_path",
        "summary_json_path",
    ):
        path = getattr(bundle.artifacts, path_field, None)
        if path is not None and path.exists():
            mlflow_module.log_artifact(str(path))
