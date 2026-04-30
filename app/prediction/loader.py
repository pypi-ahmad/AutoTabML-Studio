"""Model-loading wrappers for prediction flows."""

from __future__ import annotations

import importlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.errors import log_exception
from app.modeling.pycaret.schemas import SavedModelMetadata
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
from app.registry.errors import RegistryError, RegistryUnavailableError
from app.registry.registry_service import RegistryService
from app.security.errors import TrustedArtifactError
from app.security.trusted_artifacts import (
    load_verified_pickle_artifact,
    require_metadata_checksum,
    verify_local_artifact,
)
from app.tracking.errors import TrackingError
from app.tracking.history_service import HistoryService
from app.tracking.mlflow_query import is_mlflow_available

logger = logging.getLogger(__name__)


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

    def _trusted_metadata_dirs(self) -> list[Path]:
        return list(dict.fromkeys(self._metadata_dirs + self._model_dirs))

    def load(self, request: PredictionRequest) -> LoadedModel:
        references = self.discover()
        identifier = request.model_identifier or request.model_path
        resolved = resolve_local_model_reference(identifier, references)

        metadata = None
        if request.metadata_path is not None:
            metadata = load_saved_model_metadata_file(
                request.metadata_path,
                metadata_roots=self._trusted_metadata_dirs(),
                model_roots=self._model_dirs,
                raise_on_error=True,
            )
        elif resolved.metadata_path is not None:
            metadata = load_saved_model_metadata_file(
                resolved.metadata_path,
                metadata_roots=self._trusted_metadata_dirs(),
                model_roots=self._model_dirs,
                raise_on_error=True,
            )

        if metadata is None:
            raise ModelLoadError(
                "Local saved PyCaret models require trusted checksum-backed metadata. "
                "Re-save the model from within AutoTabML Studio before loading it."
            )

        task_type = request.task_type_hint or resolved.task_type
        task_type = coerce_prediction_task_type(metadata.task_type)

        if task_type in {None, PredictionTaskType.UNKNOWN}:
            raise ModelLoadError(
                "Local saved PyCaret models require saved metadata or an explicit task_type_hint."
            )

        try:
            from app.modeling.pycaret.persistence import load_model_artifact

            model_path = verify_local_artifact(
                metadata.model_path,
                trusted_roots=self._model_dirs,
                expected_sha256=require_metadata_checksum(metadata.model_dump(mode="json")),
                label="model artifact",
            ).path
            native_model = load_model_artifact(
                to_experiment_task_type(task_type),
                model_path,
            )
        except TrustedArtifactError as exc:
            log_exception(
                logger,
                exc,
                operation="prediction.load_local_pycaret",
                context={"model_path": str(metadata.model_path)},
            )
            raise ModelLoadError(f"Could not load local saved PyCaret model: {exc}") from exc
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            log_exception(
                logger,
                exc,
                operation="prediction.load_local_pycaret",
                context={"model_path": str(metadata.model_path)},
            )
            raise ModelLoadError(f"Could not load local saved PyCaret model: {exc}") from exc

        metadata_payload = dict(resolved.metadata)
        metadata_payload.update(
            {
                "target_column": metadata.target_column,
                "dataset_fingerprint": metadata.dataset_fingerprint,
                "dataset_name": metadata.dataset_name,
                "feature_dtypes": dict(metadata.feature_dtypes),
                "target_dtype": metadata.target_dtype,
                "model_only": metadata.model_only,
                "trained_at": metadata.trained_at,
                "artifact_format": metadata.artifact_format,
                "trusted_source": metadata.trusted_source,
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
            feature_columns=list(metadata.feature_columns),
            target_column=metadata.target_column,
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

    def _trusted_metadata_dirs(self) -> list[Path]:
        return list(dict.fromkeys(self._metadata_dirs + self._model_dirs))

    def load(self, request: PredictionRequest) -> LoadedModel:
        references = self.discover()
        identifier = request.model_identifier or request.model_path
        resolved = resolve_local_model_reference(identifier, references)

        flaml_metadata = None
        if request.metadata_path is not None:
            flaml_metadata = load_flaml_saved_model_metadata_file(
                request.metadata_path,
                metadata_roots=self._trusted_metadata_dirs(),
                model_roots=self._model_dirs,
                raise_on_error=True,
            )
        elif resolved.metadata_path is not None:
            flaml_metadata = load_flaml_saved_model_metadata_file(
                resolved.metadata_path,
                metadata_roots=self._trusted_metadata_dirs(),
                model_roots=self._model_dirs,
                raise_on_error=True,
            )

        if flaml_metadata is None:
            raise ModelLoadError(
                "FLAML models require trusted checksum-backed metadata. "
                "Re-save the model from within AutoTabML Studio before loading it."
            )

        task_type = request.task_type_hint or resolved.task_type
        if task_type in {None, PredictionTaskType.UNKNOWN}:
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

        try:
            native_model = load_verified_pickle_artifact(
                flaml_metadata.model_path,
                trusted_roots=self._model_dirs,
                expected_sha256=require_metadata_checksum(flaml_metadata.model_dump(mode="json")),
            )
        except TrustedArtifactError as exc:
            log_exception(
                logger,
                exc,
                operation="prediction.load_flaml",
                context={"model_path": str(flaml_metadata.model_path)},
            )
            raise ModelLoadError(f"Could not load FLAML model: {exc}") from exc
        except (OSError, ValueError, RuntimeError) as exc:
            log_exception(
                logger,
                exc,
                operation="prediction.load_flaml",
                context={"model_path": str(flaml_metadata.model_path)},
            )
            raise ModelLoadError(f"Could not load FLAML model: {exc}") from exc

        metadata_payload = dict(resolved.metadata)
        metadata_payload.update(
            {
                "framework": "flaml",
                "target_column": flaml_metadata.target_column,
                "dataset_name": flaml_metadata.dataset_name,
                "dataset_fingerprint": flaml_metadata.dataset_fingerprint,
                "feature_dtypes": dict(flaml_metadata.feature_dtypes),
                "target_dtype": flaml_metadata.target_dtype,
                "best_estimator": flaml_metadata.best_estimator,
                "trained_at": flaml_metadata.trained_at,
                "artifact_format": flaml_metadata.artifact_format,
                "trusted_source": flaml_metadata.trusted_source,
            }
        )

        return LoadedModel(
            source_type=ModelSourceType.LOCAL_SAVED_MODEL,
            task_type=task_type,
            model_identifier=resolved.display_name,
            load_reference=resolved.load_reference,
            loader_name=self.__class__.__name__,
            scorer_kind="flaml",
            supported_prediction_modes=[PredictionMode.SINGLE_ROW, PredictionMode.BATCH],
            feature_columns=list(flaml_metadata.feature_columns),
            target_column=flaml_metadata.target_column,
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
            metadata = self._load_optional_metadata(request.metadata_path)
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

    def _load_optional_metadata(self, path: Path) -> SavedModelMetadata | None:
        try:
            return SavedModelMetadata.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
            log_exception(
                logger,
                exc,
                operation="prediction.load_optional_metadata",
                level=logging.DEBUG,
                context={"metadata_path": str(path)},
            )
            return None

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
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            log_exception(
                logger,
                exc,
                operation="prediction.load_mlflow_pyfunc",
                context={"model_uri": model_uri},
            )
            raise ModelLoadError(f"Could not load MLflow model '{model_uri}': {exc}") from exc

    def _history_metadata(self, run_id: str) -> dict[str, Any]:
        try:
            detail = HistoryService(tracking_uri=self._tracking_uri).get_run_detail(run_id)
        except TrackingError as exc:
            log_exception(
                logger,
                exc,
                operation="prediction.history_metadata",
                level=logging.DEBUG,
                context={"run_id": run_id},
            )
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
        except (RegistryError, TrackingError) as exc:
            log_exception(
                logger,
                exc,
                operation="prediction.registry_metadata",
                context={"model_name": request.registry_model_name},
            )
            raise ModelLoadError(f"Could not resolve registered model metadata: {exc}") from exc