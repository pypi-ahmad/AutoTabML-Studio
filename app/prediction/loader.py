"""Model-loading wrappers for prediction flows."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.prediction.errors import ModelLoadError
from app.prediction.schemas import (
    AvailableModelReference,
    LoadedModel,
    ModelSourceType,
    PredictionMode,
    PredictionRequest,
    PredictionTaskType,
)
from app.prediction.selectors import (
    build_mlflow_registered_model_uri,
    build_mlflow_run_model_uri,
    coerce_prediction_task_type,
    discover_flaml_saved_models,
    discover_local_saved_models,
    extract_run_id_from_model_uri,
    load_flaml_saved_model_metadata_file,
    load_saved_model_metadata_file,
    resolve_local_model_reference,
    to_experiment_task_type,
)
from app.registry.errors import RegistryUnavailableError
from app.registry.registry_service import RegistryService
from app.tracking.history_service import HistoryService
from app.tracking.mlflow_query import is_mlflow_available


class ModelLoader(ABC):
    """Abstract model loader contract."""

    @abstractmethod
    def supports(self, source_type: ModelSourceType) -> bool:
        """Return True when the loader supports the requested source type."""

    @abstractmethod
    def load(self, request: PredictionRequest) -> LoadedModel:
        """Load and normalize a model reference for prediction."""


class LocalPyCaretModelLoader(ModelLoader):
    """Load local saved PyCaret model artifacts."""

    def __init__(self, *, model_dirs: list[Path], metadata_dirs: list[Path]) -> None:
        self._model_dirs = model_dirs
        self._metadata_dirs = metadata_dirs

    def supports(self, source_type: ModelSourceType) -> bool:
        return source_type == ModelSourceType.LOCAL_SAVED_MODEL

    def discover(self) -> list[AvailableModelReference]:
        return discover_local_saved_models(self._model_dirs, self._metadata_dirs)

    def load(self, request: PredictionRequest) -> LoadedModel:
        references = self.discover()
        identifier = request.model_identifier or request.model_path
        resolved = resolve_local_model_reference(identifier, references)

        metadata = None
        if request.metadata_path is not None:
            metadata = load_saved_model_metadata_file(request.metadata_path)
        elif resolved.metadata_path is not None:
            metadata = load_saved_model_metadata_file(resolved.metadata_path)

        task_type = request.task_type_hint or resolved.task_type
        if metadata is not None:
            task_type = coerce_prediction_task_type(metadata.task_type)

        if task_type in {None, PredictionTaskType.UNKNOWN}:
            raise ModelLoadError(
                "Local saved PyCaret models require saved metadata or an explicit task_type_hint."
            )

        try:
            from app.modeling.pycaret.persistence import load_model_artifact

            native_model = load_model_artifact(
                to_experiment_task_type(task_type),
                resolved.model_path or Path(resolved.load_reference),
            )
        except Exception as exc:
            raise ModelLoadError(f"Could not load local saved PyCaret model: {exc}") from exc

        metadata_payload = dict(resolved.metadata)
        if metadata is not None:
            metadata_payload.update(
                {
                    "target_column": metadata.target_column,
                    "dataset_fingerprint": metadata.dataset_fingerprint,
                    "feature_dtypes": dict(metadata.feature_dtypes),
                    "target_dtype": metadata.target_dtype,
                    "model_only": metadata.model_only,
                    "experiment_snapshot_path": str(metadata.experiment_snapshot_path)
                    if metadata.experiment_snapshot_path is not None
                    else None,
                }
            )

        return LoadedModel(
            source_type=ModelSourceType.LOCAL_SAVED_MODEL,
            task_type=task_type,
            model_identifier=resolved.display_name,
            load_reference=resolved.load_reference,
            loader_name=self.__class__.__name__,
            scorer_kind="pycaret",
            supported_prediction_modes=[PredictionMode.SINGLE_ROW, PredictionMode.BATCH],
            feature_columns=list(metadata.feature_columns) if metadata is not None else list(resolved.feature_columns),
            target_column=metadata.target_column if metadata is not None else resolved.metadata.get("target_column"),
            metadata=metadata_payload,
            native_model=native_model,
        )


class LocalFlamlModelLoader(ModelLoader):
    """Load local saved FLAML model artifacts."""

    def __init__(self, *, model_dirs: list[Path], metadata_dirs: list[Path]) -> None:
        self._model_dirs = model_dirs
        self._metadata_dirs = metadata_dirs

    def supports(self, source_type: ModelSourceType) -> bool:
        return source_type == ModelSourceType.LOCAL_SAVED_MODEL

    def discover(self) -> list[AvailableModelReference]:
        return discover_flaml_saved_models(self._model_dirs, self._metadata_dirs)

    def load(self, request: PredictionRequest) -> LoadedModel:
        references = self.discover()
        identifier = request.model_identifier or request.model_path
        resolved = resolve_local_model_reference(identifier, references)

        flaml_metadata = None
        if request.metadata_path is not None:
            flaml_metadata = load_flaml_saved_model_metadata_file(request.metadata_path)
        elif resolved.metadata_path is not None:
            flaml_metadata = load_flaml_saved_model_metadata_file(resolved.metadata_path)

        task_type = request.task_type_hint or resolved.task_type
        if task_type in {None, PredictionTaskType.UNKNOWN} and flaml_metadata is not None:
            raw_task = (
                flaml_metadata.task_type.value
                if hasattr(flaml_metadata.task_type, "value")
                else str(flaml_metadata.task_type)
            )
            task_type = coerce_prediction_task_type(raw_task)

        if task_type in {None, PredictionTaskType.UNKNOWN}:
            raise ModelLoadError(
                "FLAML models require saved metadata or an explicit task_type_hint."
            )

        import pickle

        model_path = resolved.model_path or Path(resolved.load_reference)
        try:
            with model_path.open("rb") as f:
                native_model = pickle.load(f)  # noqa: S301
        except Exception as exc:
            raise ModelLoadError(f"Could not load FLAML model: {exc}") from exc

        metadata_payload = dict(resolved.metadata)
        if flaml_metadata is not None:
            metadata_payload.update({
                "framework": "flaml",
                "target_column": flaml_metadata.target_column,
                "dataset_fingerprint": flaml_metadata.dataset_fingerprint,
                "feature_dtypes": dict(flaml_metadata.feature_dtypes),
                "target_dtype": flaml_metadata.target_dtype,
                "best_estimator": flaml_metadata.best_estimator,
            })

        return LoadedModel(
            source_type=ModelSourceType.LOCAL_SAVED_MODEL,
            task_type=task_type,
            model_identifier=resolved.display_name,
            load_reference=resolved.load_reference,
            loader_name=self.__class__.__name__,
            scorer_kind="flaml",
            supported_prediction_modes=[PredictionMode.SINGLE_ROW, PredictionMode.BATCH],
            feature_columns=(
                list(flaml_metadata.feature_columns)
                if flaml_metadata is not None
                else list(resolved.feature_columns)
            ),
            target_column=(
                flaml_metadata.target_column
                if flaml_metadata is not None
                else resolved.metadata.get("target_column")
            ),
            metadata=metadata_payload,
            native_model=native_model,
        )


class MLflowModelLoader(ModelLoader):
    """Load MLflow-backed pyfunc models."""

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        registry_enabled: bool = True,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._registry_uri = registry_uri
        self._registry_enabled = registry_enabled

    def supports(self, source_type: ModelSourceType) -> bool:
        return source_type in {
            ModelSourceType.MLFLOW_RUN_MODEL,
            ModelSourceType.MLFLOW_REGISTERED_MODEL,
        }

    def load(self, request: PredictionRequest) -> LoadedModel:
        model_uri = self._resolve_model_uri(request)
        resolved_load_reference = model_uri
        metadata_payload: dict[str, Any] = {"metadata_available": False, "model_uri": model_uri}
        task_type = request.task_type_hint or PredictionTaskType.UNKNOWN
        feature_columns: list[str] = []
        target_column: str | None = None
        run_id = request.run_id or extract_run_id_from_model_uri(model_uri)

        if request.metadata_path is not None:
            metadata = load_saved_model_metadata_file(request.metadata_path)
            if metadata is not None:
                task_type = coerce_prediction_task_type(metadata.task_type)
                feature_columns = list(metadata.feature_columns)
                target_column = metadata.target_column
                metadata_payload.update(
                    {
                        "metadata_available": True,
                        "target_column": metadata.target_column,
                        "dataset_fingerprint": metadata.dataset_fingerprint,
                        "feature_dtypes": dict(metadata.feature_dtypes),
                    }
                )

        if request.source_type == ModelSourceType.MLFLOW_REGISTERED_MODEL:
            registry_metadata = self._registry_metadata(request)
            metadata_payload.update(registry_metadata)
            resolved_load_reference = str(registry_metadata.get("source") or resolved_load_reference)
            metadata_payload["resolved_source"] = resolved_load_reference
            run_id = metadata_payload.get("run_id") or run_id

        native_model = self._load_pyfunc_model(resolved_load_reference)

        if run_id:
            history_metadata = self._history_metadata(run_id)
            metadata_payload.update(history_metadata)
            if task_type == PredictionTaskType.UNKNOWN:
                task_type = coerce_prediction_task_type(history_metadata.get("task_type"))
            target_column = target_column or history_metadata.get("target_column")

        model_identifier = (
            request.registry_model_name
            or request.model_identifier
            or request.model_uri
            or run_id
            or model_uri
        )

        return LoadedModel(
            source_type=request.source_type,
            task_type=task_type,
            model_identifier=str(model_identifier),
            load_reference=resolved_load_reference,
            loader_name=self.__class__.__name__,
            scorer_kind="mlflow_pyfunc",
            supported_prediction_modes=[PredictionMode.SINGLE_ROW, PredictionMode.BATCH],
            feature_columns=feature_columns,
            target_column=target_column,
            metadata=metadata_payload,
            native_model=native_model,
        )

    def _resolve_model_uri(self, request: PredictionRequest) -> str:
        if request.model_uri:
            return request.model_uri
        if request.source_type == ModelSourceType.MLFLOW_RUN_MODEL:
            return build_mlflow_run_model_uri(request.run_id or "", request.artifact_path or "")
        if request.source_type == ModelSourceType.MLFLOW_REGISTERED_MODEL:
            return build_mlflow_registered_model_uri(
                request.registry_model_name or "",
                version=request.registry_version,
                alias=request.registry_alias,
            )
        raise ModelLoadError(f"Unsupported MLflow model source: {request.source_type.value}")

    def _load_pyfunc_model(self, model_uri: str):
        if not is_mlflow_available():
            raise ModelLoadError("mlflow is not installed. Install it with: pip install mlflow")
        try:
            import mlflow
            pyfunc = importlib.import_module("mlflow.pyfunc")
            if self._tracking_uri:
                mlflow.set_tracking_uri(self._tracking_uri)
            if self._registry_uri:
                mlflow.set_registry_uri(self._registry_uri)
            return pyfunc.load_model(
                model_uri,
                dst_path=None,
            )
        except Exception as exc:
            raise ModelLoadError(f"Could not load MLflow model '{model_uri}': {exc}") from exc

    def _history_metadata(self, run_id: str) -> dict[str, Any]:
        try:
            detail = HistoryService(tracking_uri=self._tracking_uri).get_run_detail(run_id)
        except Exception:
            return {"run_id": run_id}
        return {
            "run_id": detail.run_id,
            "task_type": detail.task_type,
            "target_column": detail.target_column,
            "experiment_name": detail.experiment_name,
            "run_name": detail.run_name,
        }

    def _registry_metadata(self, request: PredictionRequest) -> dict[str, Any]:
        if not self._registry_enabled:
            return {"registry_available": False}
        if not request.registry_model_name:
            return {}
        try:
            service = RegistryService(
                tracking_uri=request.tracking_uri or self._tracking_uri,
                registry_uri=request.registry_uri or self._registry_uri,
            )
            version = None
            if request.registry_version:
                version = service.get_version(request.registry_model_name, request.registry_version)
            elif request.registry_alias:
                version = service.get_version_by_alias(request.registry_model_name, request.registry_alias)
            if version is None:
                return {"registry_model_name": request.registry_model_name}
            return {
                "registry_model_name": request.registry_model_name,
                "registry_version": version.version,
                "run_id": version.run_id,
                "source": version.source,
                "registry_status": version.app_status,
                "registry_aliases": list(version.aliases),
            }
        except RegistryUnavailableError as exc:
            raise ModelLoadError(str(exc)) from exc
        except Exception as exc:
            raise ModelLoadError(f"Could not resolve registered model metadata: {exc}") from exc