"""Streamlit cache helpers for the UI layer.

TTL strategy:
- MLflow query data is cached briefly to collapse rerun bursts while keeping
  registry and run-history views fresh.
- Dataset loads are cached longer because file and remote reads are the most
  expensive rerun path and are typically stable within a user session.
- Services and metadata stores are cached as resources for the life of the
  current Streamlit process and can be cleared manually from the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st

from app.ingestion import DatasetInputSpec, LoadedDataset, load_dataset
from app.modeling.flaml.service import FlamlAutoMLService
from app.modeling.pycaret.service import PyCaretExperimentService
from app.pages.services.experiment_workflow import ExperimentWorkflowService
from app.pages.services.prediction_workflow import PredictionWorkflowService
from app.pages.services.experiment_workflow import ExperimentWorkflowService
from app.prediction import PredictionService, SchemaValidationMode
from app.registry.registry_service import RegistryService
from app.storage.store import AppMetadataStore
from app.tracking import mlflow_query
from app.tracking.history_service import HistoryService

if TYPE_CHECKING:
    from app.config.models import AppSettings
    from app.registry.schemas import RegistryModelSummary, RegistryVersionSummary
    from app.tracking.schemas import RunHistoryItem

MLFLOW_QUERY_TTL_SECONDS = 15
DATASET_LOAD_TTL_SECONDS = 300


def load_dataset_for_ui(input_spec: DatasetInputSpec) -> LoadedDataset:
    """Load a dataset with a Streamlit cache when the input is cacheable."""

    signature = _dataset_cache_signature(input_spec)
    if signature is None:
        return load_dataset(input_spec)
    return _load_dataset_cached(signature)


def get_metadata_store(app_settings: "AppSettings") -> AppMetadataStore | None:
    """Return a cached metadata-store resource for the current app settings."""

    return _get_metadata_store_resource(database_path=str(app_settings.database.path))


def get_prediction_service(app_settings: "AppSettings") -> PredictionService:
    """Return a cached prediction service for the current UI settings."""

    prediction_settings = app_settings.prediction
    tracking_settings = app_settings.tracking
    return _get_prediction_service_resource(
        artifacts_dir=str(prediction_settings.artifacts_dir),
        history_path=str(prediction_settings.history_path),
        schema_validation_mode=prediction_settings.schema_validation_mode,
        prediction_column_name=prediction_settings.prediction_column_name,
        prediction_score_column_name=prediction_settings.prediction_score_column_name,
        local_model_dirs=tuple(str(path) for path in prediction_settings.supported_local_model_dirs),
        local_metadata_dirs=tuple(str(path) for path in prediction_settings.local_model_metadata_dirs),
        tracking_uri=tracking_settings.tracking_uri,
        registry_uri=tracking_settings.registry_uri,
        registry_enabled=tracking_settings.registry_enabled,
        database_path=str(app_settings.database.path),
    )


def get_prediction_workflow_service() -> PredictionWorkflowService:
    """Return a cached workflow helper for prediction-related pages."""

    return _get_prediction_workflow_service_resource()


def get_experiment_workflow_service() -> ExperimentWorkflowService:
    """Return a cached workflow helper for the Train & Tune page."""

    return _get_experiment_workflow_service_resource()


def get_history_service(app_settings: "AppSettings") -> HistoryService:
    """Return a cached MLflow history service for the current UI settings."""

    tracking_settings = app_settings.tracking
    return _get_history_service_resource(
        tracking_uri=tracking_settings.tracking_uri,
        default_experiment_names=tuple(tracking_settings.default_experiment_names),
        default_limit=tracking_settings.history_page_default_limit,
    )


def get_registry_service(app_settings: "AppSettings") -> RegistryService:
    """Return a cached MLflow registry service for the current UI settings."""

    tracking_settings = app_settings.tracking
    return _get_registry_service_resource(
        tracking_uri=tracking_settings.tracking_uri,
        registry_uri=tracking_settings.registry_uri,
        champion_alias=tracking_settings.champion_alias,
        candidate_alias=tracking_settings.candidate_alias,
        archived_tag_key=tracking_settings.archived_tag_key,
    )


def get_pycaret_experiment_service(app_settings: "AppSettings") -> PyCaretExperimentService:
    """Return a cached PyCaret experiment service for the current UI settings."""

    settings = app_settings.pycaret
    tracking_settings = app_settings.tracking
    return _get_pycaret_experiment_service_resource(
        artifacts_dir=str(settings.artifacts_dir),
        models_dir=str(settings.models_dir),
        snapshots_dir=str(settings.snapshots_dir),
        classification_compare_metric=settings.default_compare_metric_classification,
        regression_compare_metric=settings.default_compare_metric_regression,
        classification_tune_metric=settings.default_tune_metric_classification,
        regression_tune_metric=settings.default_tune_metric_regression,
        mlflow_experiment_name=settings.mlflow_experiment_name,
        tracking_uri=tracking_settings.tracking_uri,
        registry_uri=tracking_settings.registry_uri,
        database_path=str(app_settings.database.path),
    )


def get_flaml_automl_service(app_settings: "AppSettings") -> FlamlAutoMLService:
    """Return a cached FLAML AutoML service for the current UI settings."""

    settings = app_settings.flaml
    tracking_settings = app_settings.tracking
    return _get_flaml_automl_service_resource(
        artifacts_dir=str(settings.artifacts_dir),
        models_dir=str(settings.models_dir),
        default_classification_metric=settings.default_classification_metric,
        default_regression_metric=settings.default_regression_metric,
        mlflow_experiment_name=settings.mlflow_experiment_name,
        tracking_uri=tracking_settings.tracking_uri,
        registry_uri=tracking_settings.registry_uri,
        database_path=str(app_settings.database.path),
    )


def list_cached_mlflow_runs(
    app_settings: "AppSettings",
    *,
    limit: int | None = None,
    sort_field: str | None = None,
    sort_direction: str | None = None,
) -> list["RunHistoryItem"]:
    """Return cached MLflow run history for the current UI settings."""

    tracking_settings = app_settings.tracking
    return _list_cached_mlflow_runs(
        tracking_uri=tracking_settings.tracking_uri,
        default_experiment_names=tuple(tracking_settings.default_experiment_names),
        default_limit=tracking_settings.history_page_default_limit,
        limit=limit,
        sort_field=sort_field,
        sort_direction=sort_direction,
    )


def list_cached_registered_models(app_settings: "AppSettings") -> list["RegistryModelSummary"]:
    """Return cached registry model summaries for the current UI settings."""

    tracking_settings = app_settings.tracking
    return _list_cached_registered_models(
        tracking_uri=tracking_settings.tracking_uri,
        registry_uri=tracking_settings.registry_uri,
        champion_alias=tracking_settings.champion_alias,
        candidate_alias=tracking_settings.candidate_alias,
        archived_tag_key=tracking_settings.archived_tag_key,
    )


def list_cached_model_versions(
    app_settings: "AppSettings",
    model_name: str,
) -> list["RegistryVersionSummary"]:
    """Return cached registry versions for one registered model."""

    tracking_settings = app_settings.tracking
    return _list_cached_model_versions(
        tracking_uri=tracking_settings.tracking_uri,
        registry_uri=tracking_settings.registry_uri,
        champion_alias=tracking_settings.champion_alias,
        candidate_alias=tracking_settings.candidate_alias,
        archived_tag_key=tracking_settings.archived_tag_key,
        model_name=model_name,
    )


def invalidate_dataset_cache() -> None:
    """Clear cached dataset loads."""

    _load_dataset_cached.clear()


def invalidate_mlflow_query_cache() -> None:
    """Clear cached MLflow read models and the query-layer registry cache."""

    _list_cached_mlflow_runs.clear()
    _list_cached_registered_models.clear()
    _list_cached_model_versions.clear()
    mlflow_query.invalidate_registry_cache()


def invalidate_service_cache() -> None:
    """Clear cached service and metadata-store resources."""

    _get_metadata_store_resource.clear()
    _get_prediction_service_resource.clear()
    _get_history_service_resource.clear()
    _get_registry_service_resource.clear()
    _get_pycaret_experiment_service_resource.clear()
    _get_flaml_automl_service_resource.clear()


def invalidate_all_ui_caches() -> None:
    """Clear every UI cache layer."""

    invalidate_dataset_cache()
    invalidate_mlflow_query_cache()
    invalidate_service_cache()


def _dataset_cache_signature(input_spec: DatasetInputSpec) -> dict[str, Any] | None:
    """Return a stable cache key payload for dataset loads when possible."""

    if input_spec.dataframe is not None:
        return None

    payload = input_spec.model_dump(mode="json", exclude_none=True)
    if input_spec.path is not None:
        path = Path(input_spec.path)
        try:
            stat_result = path.stat()
        except OSError:
            path_state = None
        else:
            path_state = {
                "mtime_ns": stat_result.st_mtime_ns,
                "size": stat_result.st_size,
            }
        payload["path"] = str(path)
        payload["_path_state"] = path_state
    return payload


@st.cache_data(ttl=DATASET_LOAD_TTL_SECONDS, show_spinner=False)
def _load_dataset_cached(signature: dict[str, Any]) -> LoadedDataset:
    return load_dataset(DatasetInputSpec.model_validate(signature))


@st.cache_resource(show_spinner=False)
def _get_metadata_store_resource(*, database_path: str) -> AppMetadataStore | None:
    store = AppMetadataStore(Path(database_path))
    store.initialize_if_needed()
    return store


@st.cache_resource(show_spinner=False)
def _get_prediction_service_resource(
    *,
    artifacts_dir: str,
    history_path: str,
    schema_validation_mode: str,
    prediction_column_name: str,
    prediction_score_column_name: str,
    local_model_dirs: tuple[str, ...],
    local_metadata_dirs: tuple[str, ...],
    tracking_uri: str | None,
    registry_uri: str | None,
    registry_enabled: bool,
    database_path: str,
) -> PredictionService:
    return PredictionService(
        artifacts_dir=Path(artifacts_dir),
        history_path=Path(history_path),
        schema_validation_mode=SchemaValidationMode(schema_validation_mode),
        prediction_column_name=prediction_column_name,
        prediction_score_column_name=prediction_score_column_name,
        local_model_dirs=[Path(path) for path in local_model_dirs],
        local_metadata_dirs=[Path(path) for path in local_metadata_dirs],
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        registry_enabled=registry_enabled,
        metadata_store=_get_metadata_store_resource(database_path=database_path),
    )


@st.cache_resource(show_spinner=False)
def _get_prediction_workflow_service_resource() -> PredictionWorkflowService:
    return PredictionWorkflowService()


@st.cache_resource(show_spinner=False)
def _get_experiment_workflow_service_resource() -> ExperimentWorkflowService:
    return ExperimentWorkflowService()


@st.cache_resource(show_spinner=False)
def _get_experiment_workflow_service_resource() -> ExperimentWorkflowService:
    return ExperimentWorkflowService()


@st.cache_resource(show_spinner=False)
def _get_history_service_resource(
    *,
    tracking_uri: str | None,
    default_experiment_names: tuple[str, ...],
    default_limit: int,
) -> HistoryService:
    return HistoryService(
        tracking_uri=tracking_uri,
        default_experiment_names=list(default_experiment_names),
        default_limit=default_limit,
    )


@st.cache_resource(show_spinner=False)
def _get_registry_service_resource(
    *,
    tracking_uri: str | None,
    registry_uri: str | None,
    champion_alias: str,
    candidate_alias: str,
    archived_tag_key: str,
) -> RegistryService:
    return RegistryService(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        champion_alias=champion_alias,
        candidate_alias=candidate_alias,
        archived_tag_key=archived_tag_key,
    )


@st.cache_resource(show_spinner=False)
def _get_pycaret_experiment_service_resource(
    *,
    artifacts_dir: str,
    models_dir: str,
    snapshots_dir: str,
    classification_compare_metric: str,
    regression_compare_metric: str,
    classification_tune_metric: str,
    regression_tune_metric: str,
    mlflow_experiment_name: str,
    tracking_uri: str | None,
    registry_uri: str | None,
    database_path: str,
) -> PyCaretExperimentService:
    return PyCaretExperimentService(
        artifacts_dir=Path(artifacts_dir),
        models_dir=Path(models_dir),
        snapshots_dir=Path(snapshots_dir),
        classification_compare_metric=classification_compare_metric,
        regression_compare_metric=regression_compare_metric,
        classification_tune_metric=classification_tune_metric,
        regression_tune_metric=regression_tune_metric,
        mlflow_experiment_name=mlflow_experiment_name,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        metadata_store=_get_metadata_store_resource(database_path=database_path),
    )


@st.cache_resource(show_spinner=False)
def _get_flaml_automl_service_resource(
    *,
    artifacts_dir: str,
    models_dir: str,
    default_classification_metric: str,
    default_regression_metric: str,
    mlflow_experiment_name: str,
    tracking_uri: str | None,
    registry_uri: str | None,
    database_path: str,
) -> FlamlAutoMLService:
    return FlamlAutoMLService(
        artifacts_dir=Path(artifacts_dir),
        models_dir=Path(models_dir),
        default_classification_metric=default_classification_metric,
        default_regression_metric=default_regression_metric,
        mlflow_experiment_name=mlflow_experiment_name,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        metadata_store=_get_metadata_store_resource(database_path=database_path),
    )


@st.cache_data(ttl=MLFLOW_QUERY_TTL_SECONDS, show_spinner=False)
def _list_cached_mlflow_runs(
    *,
    tracking_uri: str | None,
    default_experiment_names: tuple[str, ...],
    default_limit: int,
    limit: int | None,
    sort_field: str | None = None,
    sort_direction: str | None = None,
) -> list["RunHistoryItem"]:
    from app.tracking.filters import RunHistorySort, RunSortField, SortDirection

    sort_spec: RunHistorySort | None = None
    if sort_field is not None:
        sort_spec = RunHistorySort(
            field=RunSortField(sort_field),
            direction=SortDirection(sort_direction or SortDirection.DESCENDING.value),
        )

    return _get_history_service_resource(
        tracking_uri=tracking_uri,
        default_experiment_names=default_experiment_names,
        default_limit=default_limit,
    ).list_runs(limit=limit, sort=sort_spec)


@st.cache_data(ttl=MLFLOW_QUERY_TTL_SECONDS, show_spinner=False)
def _list_cached_registered_models(
    *,
    tracking_uri: str | None,
    registry_uri: str | None,
    champion_alias: str,
    candidate_alias: str,
    archived_tag_key: str,
) -> list["RegistryModelSummary"]:
    return _get_registry_service_resource(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        champion_alias=champion_alias,
        candidate_alias=candidate_alias,
        archived_tag_key=archived_tag_key,
    ).list_models()


@st.cache_data(ttl=MLFLOW_QUERY_TTL_SECONDS, show_spinner=False)
def _list_cached_model_versions(
    *,
    tracking_uri: str | None,
    registry_uri: str | None,
    champion_alias: str,
    candidate_alias: str,
    archived_tag_key: str,
    model_name: str,
) -> list["RegistryVersionSummary"]:
    return _get_registry_service_resource(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        champion_alias=champion_alias,
        candidate_alias=candidate_alias,
        archived_tag_key=archived_tag_key,
    ).list_versions(model_name)