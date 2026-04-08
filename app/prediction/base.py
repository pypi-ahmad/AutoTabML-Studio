"""Prediction service facade."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.prediction.artifacts import write_prediction_artifacts
from app.prediction.batch_predict import build_batch_prediction_result, resolve_batch_dataframe
from app.prediction.history import PredictionHistoryStore
from app.prediction.loader import LocalFlamlModelLoader, LocalPyCaretModelLoader, MLflowModelLoader
from app.prediction.schemas import (
    BatchPredictionRequest,
    BatchPredictionResult,
    LoadedModel,
    ModelSourceType,
    PredictionHistoryEntry,
    PredictionMode,
    PredictionRequest,
    PredictionResult,
    PredictionStatus,
    PredictionTaskType,
    SchemaValidationMode,
    SingleRowPredictionRequest,
)
from app.prediction.scorer import PredictionScorer
from app.prediction.single_row_predict import build_single_prediction_result
from app.prediction.summary import build_history_entry, build_prediction_summary
from app.prediction.validators import (
    ensure_validation_can_score,
    normalize_single_row_input,
    validate_prediction_dataframe,
    validate_single_row_shape,
)
from app.registry.registry_service import RegistryService
from app.storage import AppJobStatus, AppJobType, AppMetadataStore, record_prediction_history_entry


class BasePredictionService(ABC):
    """Abstract prediction service contract."""

    @abstractmethod
    def discover_local_models(self):  # noqa: ANN201
        """Return discoverable local saved models."""

    @abstractmethod
    def load_model(self, request: PredictionRequest) -> LoadedModel:
        """Load a normalized model for prediction."""

    @abstractmethod
    def predict_single(self, request: SingleRowPredictionRequest) -> PredictionResult:
        """Run one-row prediction."""

    @abstractmethod
    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResult:
        """Run batch prediction."""


class PredictionService(BasePredictionService):
    """Production-style local-first prediction service."""

    def __init__(
        self,
        *,
        artifacts_dir: Path,
        history_path: Path,
        schema_validation_mode: SchemaValidationMode = SchemaValidationMode.STRICT,
        prediction_column_name: str = "prediction",
        prediction_score_column_name: str = "prediction_score",
        local_model_dirs: list[Path] | None = None,
        local_metadata_dirs: list[Path] | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        registry_enabled: bool = True,
        metadata_store: AppMetadataStore | None = None,
    ) -> None:
        self._artifacts_dir = artifacts_dir
        self._history_store = PredictionHistoryStore(history_path)
        self._metadata_store = metadata_store
        self._schema_validation_mode = schema_validation_mode
        self._prediction_column_name = prediction_column_name
        self._prediction_score_column_name = prediction_score_column_name
        self._tracking_uri = tracking_uri
        self._registry_uri = registry_uri
        self._registry_enabled = registry_enabled
        self._local_loader = LocalPyCaretModelLoader(
            model_dirs=local_model_dirs or [],
            metadata_dirs=local_metadata_dirs or [],
        )
        self._flaml_loader = LocalFlamlModelLoader(
            model_dirs=local_model_dirs or [],
            metadata_dirs=local_metadata_dirs or [],
        )
        self._mlflow_loader = MLflowModelLoader(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            registry_enabled=registry_enabled,
        )
        self._scorer = PredictionScorer()

    def discover_local_models(self):  # noqa: ANN201
        pycaret_refs = self._local_loader.discover()
        flaml_refs = self._flaml_loader.discover()
        return pycaret_refs + flaml_refs

    def list_registered_models(self):  # noqa: ANN201
        if not self._registry_enabled:
            return []
        service = RegistryService(
            tracking_uri=self._tracking_uri,
            registry_uri=self._registry_uri,
        )
        return service.list_models()

    def list_registered_model_versions(self, model_name: str):  # noqa: ANN201
        if not self._registry_enabled:
            return []
        service = RegistryService(
            tracking_uri=self._tracking_uri,
            registry_uri=self._registry_uri,
        )
        return service.list_versions(model_name)

    def list_history(self, limit: int = 20):  # noqa: ANN201
        if self._metadata_store is not None:
            records = self._metadata_store.list_recent_jobs(limit=limit, job_type=AppJobType.PREDICTION)
            entries = []
            for record in records:
                metadata = record.metadata
                entries.append(
                    PredictionHistoryEntry(
                        job_id=record.job_id,
                        timestamp=record.created_at,
                        status=(
                            PredictionStatus.SUCCESS
                            if record.status == AppJobStatus.SUCCESS
                            else PredictionStatus.FAILED
                        ),
                        mode=PredictionMode(metadata.get("mode", PredictionMode.BATCH.value)),
                        model_source=ModelSourceType(metadata.get("model_source", ModelSourceType.LOCAL_SAVED_MODEL.value)),
                        model_identifier=metadata.get("model_identifier", record.title or record.job_id),
                        task_type=PredictionTaskType(metadata.get("task_type", PredictionTaskType.UNKNOWN.value)),
                        input_source=record.dataset_name or metadata.get("input_source", "prediction_input"),
                        row_count=int(metadata.get("row_count", 0)),
                        output_artifact_path=record.primary_artifact_path,
                        summary_json_path=record.summary_path,
                        metadata_json_path=(
                            Path(metadata["metadata_json_path"]) if metadata.get("metadata_json_path") else None
                        ),
                    )
                )
            return entries
        return self._history_store.list_recent(limit)

    def load_model(self, request: PredictionRequest) -> LoadedModel:
        if request.source_type == ModelSourceType.LOCAL_SAVED_MODEL:
            # Check if this is a FLAML model by looking for a FLAML metadata sidecar
            flaml_refs = self._flaml_loader.discover()
            identifier = str(request.model_identifier or request.model_path or "")
            for ref in flaml_refs:
                if identifier and identifier.lower() in {
                    ref.model_identifier.lower(),
                    ref.display_name.lower(),
                    ref.load_reference.lower(),
                }:
                    return self._flaml_loader.load(request)
            return self._local_loader.load(request)
        return self._mlflow_loader.load(request)

    def predict_single(self, request: SingleRowPredictionRequest) -> PredictionResult:
        input_source = request.input_source_label or "manual_row"
        try:
            loaded_model = self.load_model(request)
            raw_frame = normalize_single_row_input(request.row_data)
            validate_single_row_shape(raw_frame)

            validation_mode = self._resolve_validation_mode(request)
            normalized_frame, validation = validate_prediction_dataframe(
                raw_frame,
                loaded_model,
                validation_mode=validation_mode,
            )
            ensure_validation_can_score(validation)

            scored_features = self._scorer.score(
                loaded_model,
                normalized_frame,
                prediction_column_name=self._prediction_column_name,
                prediction_score_column_name=self._prediction_score_column_name,
            )
            scored = raw_frame.copy()
            scored[self._prediction_column_name] = scored_features[self._prediction_column_name].to_list()
            if self._prediction_score_column_name in scored_features.columns:
                scored[self._prediction_score_column_name] = scored_features[
                    self._prediction_score_column_name
                ].to_list()

            warnings = list(validation.warnings)
            summary = build_prediction_summary(
                mode=PredictionMode.SINGLE_ROW,
                loaded_model=loaded_model,
                input_source=input_source,
                input_row_count=1,
                rows_scored=1,
                rows_failed=0,
                prediction_column=self._prediction_column_name,
                prediction_score_column=(
                    self._prediction_score_column_name
                    if self._prediction_score_column_name in scored.columns
                    else None
                ),
                validation_mode=validation_mode,
                warnings=warnings,
            )

            artifacts = write_prediction_artifacts(
                loaded_model=loaded_model,
                scored_dataframe=scored,
                summary=summary,
                output_dir=request.output_dir or self._artifacts_dir,
                output_stem=request.output_stem,
            )
            summary.output_artifact_path = artifacts.scored_csv_path
            history_entry = build_history_entry(
                summary,
                summary_json_path=artifacts.summary_json_path,
                metadata_json_path=artifacts.metadata_json_path,
            )
            history_warning = self._persist_history(history_entry)
            if history_warning is not None:
                warnings.append(history_warning)
                summary.warnings.append(history_warning)

            return build_single_prediction_result(
                loaded_model=loaded_model,
                scored_dataframe=scored,
                validation=validation,
                summary=summary,
                artifacts=artifacts,
                history_entry=history_entry,
                warnings=warnings,
                prediction_column_name=self._prediction_column_name,
                prediction_score_column_name=self._prediction_score_column_name,
            )
        except Exception as exc:
            self._record_failure(
                request=request,
                mode=PredictionMode.SINGLE_ROW,
                input_source=input_source,
                row_count=1,
            )
            raise exc

    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResult:
        dataframe = None
        try:
            dataframe, input_source = resolve_batch_dataframe(request)
            loaded_model = self.load_model(request)
            validation_mode = self._resolve_validation_mode(request)
            normalized_frame, validation = validate_prediction_dataframe(
                dataframe,
                loaded_model,
                validation_mode=validation_mode,
            )
            ensure_validation_can_score(validation)

            scored_features = self._scorer.score(
                loaded_model,
                normalized_frame,
                prediction_column_name=self._prediction_column_name,
                prediction_score_column_name=self._prediction_score_column_name,
            )
            scored = dataframe.copy()
            scored[self._prediction_column_name] = scored_features[self._prediction_column_name].to_list()
            if self._prediction_score_column_name in scored_features.columns:
                scored[self._prediction_score_column_name] = scored_features[
                    self._prediction_score_column_name
                ].to_list()

            warnings = list(validation.warnings)
            summary = build_prediction_summary(
                mode=PredictionMode.BATCH,
                loaded_model=loaded_model,
                input_source=input_source,
                input_row_count=len(normalized_frame.index),
                rows_scored=len(scored.index),
                rows_failed=0,
                prediction_column=self._prediction_column_name,
                prediction_score_column=(
                    self._prediction_score_column_name
                    if self._prediction_score_column_name in scored.columns
                    else None
                ),
                validation_mode=validation_mode,
                warnings=warnings,
            )

            artifacts = write_prediction_artifacts(
                loaded_model=loaded_model,
                scored_dataframe=scored,
                summary=summary,
                output_dir=request.output_dir or self._artifacts_dir,
                output_path=request.output_path,
                output_stem=request.output_stem,
            )
            summary.output_artifact_path = artifacts.scored_csv_path
            history_entry = build_history_entry(
                summary,
                summary_json_path=artifacts.summary_json_path,
                metadata_json_path=artifacts.metadata_json_path,
            )
            history_warning = self._persist_history(history_entry)
            if history_warning is not None:
                warnings.append(history_warning)
                summary.warnings.append(history_warning)

            return build_batch_prediction_result(
                loaded_model=loaded_model,
                scored_dataframe=scored,
                validation=validation,
                summary=summary,
                artifacts=artifacts,
                history_entry=history_entry,
                warnings=warnings,
            )
        except Exception as exc:
            self._record_failure(
                request=request,
                mode=PredictionMode.BATCH,
                input_source=request.input_source_label or request.dataset_name or "batch_input",
                row_count=len(dataframe.index) if dataframe is not None else 0,
            )
            raise exc

    def _persist_history(self, entry):  # noqa: ANN001, ANN201
        try:
            if self._metadata_store is not None:
                record_prediction_history_entry(self._metadata_store, entry)
                return None
            self._history_store.append(entry)
            return None
        except Exception as exc:
            return f"Prediction history could not be written: {exc}"

    def _record_failure(
        self,
        *,
        request: PredictionRequest,
        mode: PredictionMode,
        input_source: str,
        row_count: int,
    ) -> None:
        summary = build_prediction_summary(
            mode=mode,
            loaded_model=LoadedModel(
                source_type=request.source_type,
                task_type=request.task_type_hint or PredictionTaskType.UNKNOWN,
                model_identifier=self._request_model_identifier(request),
                load_reference=request.model_uri or str(request.model_path or request.model_identifier or ""),
                loader_name="unknown",
                scorer_kind="unknown",
                supported_prediction_modes=[],
                native_model=None,
            ),
            input_source=input_source,
            input_row_count=row_count,
            rows_scored=0,
            rows_failed=row_count,
            prediction_column=self._prediction_column_name,
            prediction_score_column=self._prediction_score_column_name,
            validation_mode=self._resolve_validation_mode(request),
            warnings=[],
            status=PredictionStatus.FAILED,
        )
        history_entry = build_history_entry(summary)
        self._persist_history(history_entry)

    def _resolve_validation_mode(self, request: PredictionRequest) -> SchemaValidationMode:
        return request.schema_validation_mode or self._schema_validation_mode

    def _request_model_identifier(self, request: PredictionRequest) -> str:
        return (
            request.registry_model_name
            or request.model_identifier
            or request.model_uri
            or str(request.model_path or request.run_id or "unknown_model")
        )