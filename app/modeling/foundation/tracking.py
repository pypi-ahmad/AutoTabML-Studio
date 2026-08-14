"""Minimal MLflow summary logging for foundation-model runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def log_foundation_run(
    *,
    run_type: str,
    dataset_name: str,
    params: dict[str, Any],
    metrics: dict[str, float | int],
    summary_path: Path,
    tracking_uri: str | None = None,
) -> tuple[str | None, str | None]:
    """Log aggregate configuration, metrics, and the non-sensitive summary artifact."""

    try:
        import mlflow
    except ImportError:
        return None, "MLflow tracking skipped because mlflow is not installed."
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("AutoTabML Foundation Models")
        with mlflow.start_run(run_name=f"{run_type}-{dataset_name}") as run:
            mlflow.set_tags({"framework": run_type, "autotabml.run_type": run_type})
            mlflow.log_params({key: str(value) for key, value in params.items() if value is not None})
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
            if summary_path.is_file():
                mlflow.log_artifact(str(summary_path))
            return run.info.run_id, None
    except Exception as exc:  # pragma: no cover - optional integration boundary
        return None, f"MLflow tracking failed: {exc}"


__all__ = ["log_foundation_run"]
