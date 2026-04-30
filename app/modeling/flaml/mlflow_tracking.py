"""Explicit MLflow tracking for FLAML AutoML runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.modeling.base import (
    BaseTracker,
)
from app.modeling.base import (
    get_mlflow_module as _base_get_mlflow_module,
)
from app.modeling.base import (
    is_mlflow_available as _base_is_mlflow_available,
)
from app.modeling.base import (
    mlflow_exception_types as _base_mlflow_exception_types,
)
from app.modeling.flaml.schemas import FlamlResultBundle

logger = logging.getLogger(__name__)


def is_mlflow_available() -> bool:
    """Return True when mlflow is importable."""

    return _base_is_mlflow_available()


def _get_mlflow_module() -> Any:
    return _base_get_mlflow_module()


def _mlflow_exception_types(mlflow_module: Any) -> tuple[type[BaseException], ...]:
    return _base_mlflow_exception_types(mlflow_module)


class MLflowFlamlTracker(BaseTracker[FlamlResultBundle]):
    """Lightweight explicit MLflow tracker for FLAML runs."""

    def log_flaml_bundle(
        self,
        bundle: FlamlResultBundle,
        *,
        existing_run_id: str | None = None,
    ) -> tuple[str | None, list[str]]:
        """Backward-compatible FLAML tracker entrypoint."""

        return self.log_bundle(bundle, existing_run_id=existing_run_id)

    def _is_mlflow_available(self) -> bool:
        return is_mlflow_available()

    def _get_mlflow_module(self) -> Any:
        return _get_mlflow_module()

    def _operation_name(self) -> str:
        return "flaml.mlflow_tracking"

    def _build_run_name(self, bundle: FlamlResultBundle) -> str:
        return _build_run_name(bundle)

    def _build_params(self, bundle: FlamlResultBundle) -> dict[str, Any]:
        return _build_params(bundle)

    def _build_metrics(self, bundle: FlamlResultBundle) -> dict[str, float]:
        return _build_metrics(bundle)

    def _artifact_paths(self, bundle: FlamlResultBundle) -> list[Path | None]:
        artifacts = bundle.artifacts
        if artifacts is None:
            return []
        return [
            artifacts.search_result_json_path,
            artifacts.leaderboard_csv_path,
            artifacts.leaderboard_json_path,
            artifacts.summary_json_path,
            artifacts.saved_model_metadata_path,
        ]


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
