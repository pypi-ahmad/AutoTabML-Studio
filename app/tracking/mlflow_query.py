"""Clean wrappers around MLflow client operations.

All raw MLflow interaction is centralized here so that business code
never imports mlflow directly.  Every public method returns app-level
schemas and raises app-level exceptions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.registry.errors import ModelNotFoundError, RegistryUnavailableError, VersionNotFoundError
from app.registry.schemas import RegistryModelSummary, RegistryVersionSummary
from app.tracking.errors import ExperimentNotFoundError, RunNotFoundError, TrackingUnavailableError
from app.tracking.schemas import ExperimentInfo, RunDetailView, RunHistoryItem, RunStatus, RunType

logger = logging.getLogger(__name__)

# Well-known experiment names used by existing trackers.
_BENCHMARK_EXPERIMENT_PREFIX = "autotabml-benchmark"
_EXPERIMENT_EXPERIMENT_PREFIX = "autotabml-experiment"


def is_mlflow_available() -> bool:
    """Return True when mlflow is importable."""

    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


def _require_mlflow():
    if not is_mlflow_available():
        raise TrackingUnavailableError(
            "mlflow is not installed. Install it with: pip install mlflow"
        )


def _get_client(tracking_uri: str | None = None, registry_uri: str | None = None):
    """Return an MlflowClient configured with the given URIs."""

    _require_mlflow()
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)


def _ms_to_datetime(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Experiment queries
# ---------------------------------------------------------------------------


def list_experiments(
    *,
    tracking_uri: str | None = None,
) -> list[ExperimentInfo]:
    """Return all active MLflow experiments."""

    client = _get_client(tracking_uri)
    raw_experiments = client.search_experiments(view_type=1)  # ACTIVE_ONLY
    return [_normalize_experiment(exp) for exp in raw_experiments]


def get_experiment_by_name(
    name: str,
    *,
    tracking_uri: str | None = None,
) -> ExperimentInfo:
    """Return one experiment by name or raise ExperimentNotFoundError."""

    client = _get_client(tracking_uri)
    exp = client.get_experiment_by_name(name)
    if exp is None:
        raise ExperimentNotFoundError(f"Experiment '{name}' not found.")
    return _normalize_experiment(exp)


# ---------------------------------------------------------------------------
# Run queries
# ---------------------------------------------------------------------------


def search_runs(
    *,
    experiment_ids: list[str] | None = None,
    filter_string: str = "",
    order_by: list[str] | None = None,
    max_results: int = 200,
    tracking_uri: str | None = None,
    experiment_name_map: dict[str, str] | None = None,
) -> list[RunHistoryItem]:
    """Search MLflow runs and return normalized RunHistoryItem list."""

    client = _get_client(tracking_uri)
    if experiment_ids is None:
        experiments = client.search_experiments(view_type=1)
        experiment_ids = [exp.experiment_id for exp in experiments]
    if not experiment_ids:
        return []

    name_map = experiment_name_map or {}
    raw_runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        order_by=order_by or ["attributes.start_time DESC"],
        max_results=max_results,
    )
    return [_normalize_run(run, name_map) for run in raw_runs]


def get_run(
    run_id: str,
    *,
    tracking_uri: str | None = None,
    experiment_name_map: dict[str, str] | None = None,
) -> RunDetailView:
    """Fetch a single run by id and return an extended detail view."""

    client = _get_client(tracking_uri)
    try:
        raw_run = client.get_run(run_id)
    except Exception as exc:
        raise RunNotFoundError(f"Run '{run_id}' not found: {exc}") from exc

    name_map = experiment_name_map or {}
    item = _normalize_run(raw_run, name_map)

    artifacts: list[str] = []
    try:
        artifacts = _list_artifact_paths(client, run_id)
    except Exception:  # pragma: no cover
        pass

    return RunDetailView(
        **item.model_dump(),
        artifact_paths=artifacts,
    )


# ---------------------------------------------------------------------------
# Registry read queries
# ---------------------------------------------------------------------------


def list_registered_models(
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> list[RegistryModelSummary]:
    """Return all registered models from the MLflow model registry."""

    client = _get_client(tracking_uri, registry_uri)
    try:
        raw_models = client.search_registered_models()
    except Exception as exc:
        raise RegistryUnavailableError(
            f"Model registry is not available. This MLflow backend may not expose "
            f"registry APIs. Error: {exc}"
        ) from exc
    return [
        _normalize_registered_model(
            m,
            versions=_try_list_model_versions(client, m.name),
        )
        for m in raw_models
    ]


def get_registered_model(
    name: str,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> RegistryModelSummary:
    """Return a single registered model by name."""

    client = _get_client(tracking_uri, registry_uri)
    try:
        raw = client.get_registered_model(name)
    except Exception as exc:
        raise ModelNotFoundError(f"Registered model '{name}' not found: {exc}") from exc
    return _normalize_registered_model(raw, versions=_try_list_model_versions(client, name))


def list_model_versions(
    name: str,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> list[RegistryVersionSummary]:
    """Return all versions for a registered model."""

    client = _get_client(tracking_uri, registry_uri)
    try:
        raw_versions = client.search_model_versions(f"name='{name}'")
    except Exception as exc:
        raise ModelNotFoundError(
            f"Could not list versions for model '{name}': {exc}"
        ) from exc
    return [_normalize_model_version(v) for v in raw_versions]


def get_model_version(
    name: str,
    version: str,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> RegistryVersionSummary:
    """Return a single model version."""

    client = _get_client(tracking_uri, registry_uri)
    try:
        raw = client.get_model_version(name, version)
    except Exception as exc:
        raise VersionNotFoundError(
            f"Version '{version}' of model '{name}' not found: {exc}"
        ) from exc
    return _normalize_model_version(raw)


def get_model_version_by_alias(
    name: str,
    alias: str,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> RegistryVersionSummary:
    """Return the model version currently assigned to an alias."""

    client = _get_client(tracking_uri, registry_uri)
    try:
        raw = client.get_model_version_by_alias(name, alias)
    except Exception as exc:
        raise VersionNotFoundError(
            f"Alias '{alias}' of model '{name}' not found: {exc}"
        ) from exc
    return _normalize_model_version(raw)


# ---------------------------------------------------------------------------
# Registry write operations
# ---------------------------------------------------------------------------


def create_registered_model(
    name: str,
    *,
    description: str = "",
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> RegistryModelSummary:
    """Create a new registered model."""

    client = _get_client(tracking_uri, registry_uri)
    try:
        raw = client.create_registered_model(
            name,
            description=description,
            tags=tags or {},
        )
    except Exception as exc:
        raise RegistryUnavailableError(
            f"Could not create registered model '{name}': {exc}"
        ) from exc
    return _normalize_registered_model(raw)


def create_model_version(
    name: str,
    *,
    source: str,
    run_id: str | None = None,
    description: str = "",
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> RegistryVersionSummary:
    """Create a new model version under a registered model."""

    client = _get_client(tracking_uri, registry_uri)
    try:
        raw = client.create_model_version(
            name,
            source=source,
            run_id=run_id,
            description=description,
            tags=tags or {},
        )
    except Exception as exc:
        raise RegistryUnavailableError(
            f"Could not create version for model '{name}': {exc}"
        ) from exc
    return _normalize_model_version(raw)


def set_model_alias(
    name: str,
    alias: str,
    version: str,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> None:
    """Assign an alias to a specific model version."""

    client = _get_client(tracking_uri, registry_uri)
    client.set_registered_model_alias(name, alias, version)


def delete_model_alias(
    name: str,
    alias: str,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> None:
    """Remove an alias from a registered model."""

    client = _get_client(tracking_uri, registry_uri)
    try:
        client.delete_registered_model_alias(name, alias)
    except Exception:
        pass  # Alias may not exist; treat as idempotent.


def delete_model_version_tag(
    name: str,
    version: str,
    key: str,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> None:
    """Remove a tag from a specific model version."""

    client = _get_client(tracking_uri, registry_uri)
    try:
        client.delete_model_version_tag(name, version, key)
    except Exception:
        pass  # Missing tags should be treated as idempotent cleanup.


def set_model_tag(
    name: str,
    key: str,
    value: str,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> None:
    """Set a tag on a registered model."""

    client = _get_client(tracking_uri, registry_uri)
    client.set_registered_model_tag(name, key, value)


def set_model_version_tag(
    name: str,
    version: str,
    key: str,
    value: str,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> None:
    """Set a tag on a specific model version."""

    client = _get_client(tracking_uri, registry_uri)
    client.set_model_version_tag(name, version, key, value)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _normalize_experiment(raw) -> ExperimentInfo:
    tags = {}
    if hasattr(raw, "tags") and raw.tags:
        tags = dict(raw.tags) if isinstance(raw.tags, dict) else {t.key: t.value for t in raw.tags}

    return ExperimentInfo(
        experiment_id=raw.experiment_id,
        name=raw.name,
        lifecycle_stage=getattr(raw, "lifecycle_stage", "active"),
        artifact_location=getattr(raw, "artifact_location", None),
        creation_time=_ms_to_datetime(getattr(raw, "creation_time", None)),
        last_update_time=_ms_to_datetime(getattr(raw, "last_update_time", None)),
        tags=tags,
    )


def _normalize_run(raw, experiment_name_map: dict[str, str]) -> RunHistoryItem:
    info = raw.info
    data = raw.data

    params: dict[str, str] = dict(data.params) if data.params else {}
    metrics: dict[str, float] = dict(data.metrics) if data.metrics else {}
    tags: dict[str, str] = dict(data.tags) if data.tags else {}

    experiment_name = experiment_name_map.get(info.experiment_id)
    run_name = getattr(info, "run_name", None) or tags.get("mlflow.runName")
    run_type = _infer_run_type(experiment_name, tags, run_name=run_name)

    start = _ms_to_datetime(info.start_time)
    end = _ms_to_datetime(info.end_time)
    duration = None
    if start and end:
        duration = round((end - start).total_seconds(), 4)

    status = _safe_run_status(info.status)

    model_name = (
        params.get("best_baseline_model_name")
        or params.get("tuned_model_name")
        or params.get("selected_model_name")
    )

    primary_metric_name, primary_metric_value = _extract_primary_metric(
        metrics, params, run_type,
    )

    return RunHistoryItem(
        run_id=info.run_id,
        experiment_id=info.experiment_id,
        experiment_name=experiment_name,
        run_name=run_name,
        status=status,
        start_time=start,
        end_time=end,
        duration_seconds=duration,
        artifact_uri=info.artifact_uri,
        run_type=run_type,
        task_type=params.get("task_type"),
        target_column=params.get("target_column"),
        dataset_name=_extract_dataset_name(params, tags, run_name=run_name),
        dataset_fingerprint=params.get("dataset_fingerprint") or None,
        model_name=model_name,
        primary_metric_name=primary_metric_name,
        primary_metric_value=primary_metric_value,
        params=params,
        metrics=metrics,
        tags=tags,
    )


def _infer_run_type(
    experiment_name: str | None,
    tags: dict[str, str],
    *,
    run_name: str | None = None,
) -> RunType:
    if experiment_name:
        low = experiment_name.lower()
        if _BENCHMARK_EXPERIMENT_PREFIX in low:
            return RunType.BENCHMARK
        if _EXPERIMENT_EXPERIMENT_PREFIX in low:
            return RunType.EXPERIMENT
    effective_run_name = (run_name or tags.get("mlflow.runName", "")).strip()
    if effective_run_name.startswith("benchmark-"):
        return RunType.BENCHMARK
    if effective_run_name.startswith("experiment-"):
        return RunType.EXPERIMENT
    return RunType.UNKNOWN


def _safe_run_status(raw_status: str | None) -> RunStatus:
    if raw_status is None:
        return RunStatus.UNKNOWN
    try:
        return RunStatus(raw_status)
    except ValueError:
        return RunStatus.UNKNOWN


def _extract_primary_metric(
    metrics: dict[str, float],
    params: dict[str, str],
    run_type: RunType,
) -> tuple[str | None, float | None]:
    candidates = [
        params.get("compare_optimize_metric"),
        params.get("ranking_metric"),
        params.get("tune_optimize_metric"),
    ]
    for name in candidates:
        if name and name in metrics:
            return name, metrics[name]

    score_keys = ["best_baseline_score", "tuned_score", "best_score"]
    for key in score_keys:
        if key in metrics:
            return key, metrics[key]

    return None, None


def _normalize_registered_model(raw, versions: list[Any] | None = None) -> RegistryModelSummary:
    tags = _extract_tags(raw)
    aliases: dict[str, str] = {}
    if hasattr(raw, "aliases") and raw.aliases:
        if isinstance(raw.aliases, dict):
            aliases = {
                str(alias_name): str(alias_version)
                for alias_name, alias_version in raw.aliases.items()
            }
        else:
            for alias in raw.aliases:
                if hasattr(alias, "alias") and hasattr(alias, "version"):
                    aliases[str(alias.alias)] = str(alias.version)

    latest_versions = versions if versions is not None else (getattr(raw, "latest_versions", None) or [])
    version_count = len(latest_versions)
    latest_version = None
    if latest_versions:
        latest_version = str(max(_safe_version_number(v.version) for v in latest_versions))

    return RegistryModelSummary(
        name=raw.name,
        creation_timestamp=_ms_to_datetime(getattr(raw, "creation_timestamp", None)),
        last_updated_timestamp=_ms_to_datetime(getattr(raw, "last_updated_timestamp", None)),
        description=getattr(raw, "description", "") or "",
        tags=tags,
        aliases=aliases,
        version_count=version_count,
        latest_version=latest_version,
    )


def _normalize_model_version(raw) -> RegistryVersionSummary:
    tags = _extract_tags(raw)
    aliases = []
    if hasattr(raw, "aliases") and raw.aliases:
        aliases = list(raw.aliases) if isinstance(raw.aliases, (list, tuple, set)) else []

    app_status = tags.get("app.status")

    return RegistryVersionSummary(
        model_name=raw.name,
        version=str(raw.version),
        creation_timestamp=_ms_to_datetime(getattr(raw, "creation_timestamp", None)),
        last_updated_timestamp=_ms_to_datetime(getattr(raw, "last_updated_timestamp", None)),
        description=getattr(raw, "description", "") or "",
        source=getattr(raw, "source", None),
        run_id=getattr(raw, "run_id", None),
        run_link=getattr(raw, "run_link", None) or None,
        status=getattr(raw, "status", "UNKNOWN"),
        tags=tags,
        aliases=aliases,
        app_status=app_status,
    )


def _extract_tags(raw) -> dict[str, str]:
    tags_raw = getattr(raw, "tags", None)
    if tags_raw is None:
        return {}
    if isinstance(tags_raw, dict):
        return dict(tags_raw)
    result = {}
    for tag in tags_raw:
        if hasattr(tag, "key") and hasattr(tag, "value"):
            result[tag.key] = tag.value
    return result


def _list_artifact_paths(client, run_id: str, path: str | None = None) -> list[str]:
    entries = client.list_artifacts(run_id, path=path)
    paths: list[str] = []
    for entry in entries:
        entry_path = getattr(entry, "path", None)
        if not entry_path:
            continue
        if getattr(entry, "is_dir", False):
            paths.extend(_list_artifact_paths(client, run_id, path=entry_path))
        else:
            paths.append(entry_path)
    return paths


def _extract_dataset_name(
    params: dict[str, str],
    tags: dict[str, str],
    *,
    run_name: str | None = None,
) -> str | None:
    for candidate in (
        params.get("dataset_name"),
        tags.get("dataset_name"),
        tags.get("app.dataset_name"),
    ):
        if candidate:
            normalized = candidate.strip()
            if normalized:
                return normalized

    effective_run_name = (run_name or tags.get("mlflow.runName", "")).strip()
    if not effective_run_name:
        return None

    parts = effective_run_name.split("-")
    if len(parts) >= 3 and parts[0] in {"benchmark", "experiment"}:
        dataset_name = "-".join(parts[2:]).strip()
        return dataset_name or None
    return None


def _try_list_model_versions(client, name: str) -> list[Any] | None:
    try:
        return list(client.search_model_versions(f"name='{name}'"))
    except Exception:
        return None


def _safe_version_number(version: Any) -> int:
    try:
        return int(str(version))
    except (TypeError, ValueError):
        return 0
