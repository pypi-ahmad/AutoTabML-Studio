"""Schemas for prediction / inference workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ingestion.schemas import DatasetInputSpec


class ModelSourceType(str, Enum):
    """Supported model source types for prediction."""

    LOCAL_SAVED_MODEL = "local_saved_model"
    MLFLOW_RUN_MODEL = "mlflow_run_model"
    MLFLOW_REGISTERED_MODEL = "mlflow_registered_model"


class PredictionTaskType(str, Enum):
    """Task types supported during prediction."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class PredictionMode(str, Enum):
    """Prediction execution modes."""

    SINGLE_ROW = "single_row"
    BATCH = "batch"


class SchemaValidationMode(str, Enum):
    """How prediction-time schema differences should be handled."""

    STRICT = "strict"
    WARN = "warn"


class PredictionStatus(str, Enum):
    """Execution status for prediction jobs."""

    SUCCESS = "success"
    FAILED = "failed"


class PredictionInputSourceType(str, Enum):
    """Supported input origins for prediction jobs."""

    MANUAL_ROW = "manual_row"
    DATAFRAME = "dataframe"
    FILE = "file"
    SESSION_DATASET = "session_dataset"


class PredictionValidationSeverity(str, Enum):
    """Severity levels for prediction validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class AvailableModelReference(BaseModel):
    """Normalized model reference shown in discovery and selection UIs."""

    source_type: ModelSourceType
    display_name: str
    model_identifier: str
    load_reference: str
    task_type: PredictionTaskType = PredictionTaskType.UNKNOWN
    description: str = ""
    model_path: Path | None = None
    metadata_path: Path | None = None
    registry_model_name: str | None = None
    registry_version: str | None = None
    registry_alias: str | None = None
    run_id: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoadedModel(BaseModel):
    """Normalized loaded-model wrapper used across scorers and services."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_type: ModelSourceType
    task_type: PredictionTaskType = PredictionTaskType.UNKNOWN
    model_identifier: str
    load_reference: str
    loader_name: str
    scorer_kind: str
    supported_prediction_modes: list[PredictionMode] = Field(default_factory=list)
    feature_columns: list[str] = Field(default_factory=list)
    target_column: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    native_model: Any = Field(exclude=True)


class PredictionValidationIssue(BaseModel):
    """One validation issue found before scoring."""

    severity: PredictionValidationSeverity
    message: str
    field_name: str | None = None


class PredictionValidationResult(BaseModel):
    """Validation outcome for a prediction request."""

    can_score: bool = True
    metadata_available: bool = False
    missing_columns: list[str] = Field(default_factory=list)
    unexpected_columns: list[str] = Field(default_factory=list)
    normalized_columns: list[str] = Field(default_factory=list)
    issues: list[PredictionValidationIssue] = Field(default_factory=list)

    @property
    def warnings(self) -> list[str]:
        return [issue.message for issue in self.issues if issue.severity == PredictionValidationSeverity.WARNING]

    @property
    def errors(self) -> list[str]:
        return [issue.message for issue in self.issues if issue.severity == PredictionValidationSeverity.ERROR]


class PredictionRequest(BaseModel):
    """Base request for loading a model and running inference."""

    source_type: ModelSourceType
    model_identifier: str | None = None
    model_path: Path | None = None
    model_uri: str | None = None
    metadata_path: Path | None = None
    task_type_hint: PredictionTaskType | None = None
    schema_validation_mode: SchemaValidationMode | None = None
    tracking_uri: str | None = None
    registry_uri: str | None = None
    run_id: str | None = None
    artifact_path: str | None = None
    registry_model_name: str | None = None
    registry_version: str | None = None
    registry_alias: str | None = None
    input_source_label: str | None = None
    output_dir: Path | None = None
    output_stem: str | None = None

    @model_validator(mode="after")
    def validate_source_fields(self) -> "PredictionRequest":
        if self.source_type == ModelSourceType.LOCAL_SAVED_MODEL:
            if self.model_path is None and not self.model_identifier:
                raise ValueError("Local saved-model prediction requires a model_path or model_identifier.")

        if self.source_type == ModelSourceType.MLFLOW_RUN_MODEL:
            has_uri = bool(self.model_uri)
            has_run_reference = bool(self.run_id and self.artifact_path)
            if not has_uri and not has_run_reference:
                raise ValueError(
                    "MLflow run-model prediction requires model_uri or both run_id and artifact_path."
                )

        if self.source_type == ModelSourceType.MLFLOW_REGISTERED_MODEL:
            has_uri = bool(self.model_uri)
            has_model_selector = bool(self.registry_model_name and (self.registry_version or self.registry_alias))
            if not has_uri and not has_model_selector:
                raise ValueError(
                    "MLflow registered-model prediction requires model_uri or registry_model_name with registry_version or registry_alias."
                )
            if self.registry_version and self.registry_alias:
                raise ValueError("Specify either registry_version or registry_alias, not both.")

        return self


class SingleRowPredictionRequest(PredictionRequest):
    """Single-row prediction request."""

    row_data: dict[str, Any]


class BatchPredictionRequest(PredictionRequest):
    """Batch prediction request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: pd.DataFrame | None = None
    input_spec: DatasetInputSpec | None = None
    input_source_type: PredictionInputSourceType = PredictionInputSourceType.DATAFRAME
    dataset_name: str | None = None
    output_path: Path | None = None

    @model_validator(mode="after")
    def validate_batch_input(self) -> "BatchPredictionRequest":
        if self.dataframe is None and self.input_spec is None:
            raise ValueError("Batch prediction requires a dataframe or an ingestion input_spec.")
        return self


class PredictionArtifactBundle(BaseModel):
    """Artifact paths emitted for one prediction job."""

    scored_csv_path: Path | None = None
    summary_json_path: Path | None = None
    metadata_json_path: Path | None = None
    markdown_summary_path: Path | None = None


class PredictionSummary(BaseModel):
    """Roll-up summary for a prediction job."""

    run_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: PredictionStatus = PredictionStatus.SUCCESS
    mode: PredictionMode
    source_type: ModelSourceType
    task_type: PredictionTaskType = PredictionTaskType.UNKNOWN
    model_identifier: str
    input_source: str
    input_row_count: int
    rows_scored: int
    rows_failed: int = 0
    output_artifact_path: Path | None = None
    prediction_column: str = "prediction"
    prediction_score_column: str | None = None
    validation_mode: SchemaValidationMode = SchemaValidationMode.STRICT
    warnings: list[str] = Field(default_factory=list)


class PredictionHistoryEntry(BaseModel):
    """Persisted history record for a prediction job."""

    job_id: str
    timestamp: datetime
    status: PredictionStatus
    mode: PredictionMode
    model_source: ModelSourceType
    model_identifier: str
    task_type: PredictionTaskType = PredictionTaskType.UNKNOWN
    input_source: str
    row_count: int
    output_artifact_path: Path | None = None
    summary_json_path: Path | None = None
    metadata_json_path: Path | None = None


class PredictionResult(BaseModel):
    """Normalized result for a single-row prediction."""

    loaded_model: LoadedModel
    validation: PredictionValidationResult
    summary: PredictionSummary
    artifacts: PredictionArtifactBundle | None = None
    history_entry: PredictionHistoryEntry | None = None
    warnings: list[str] = Field(default_factory=list)
    row_index: Any = None
    predicted_label: Any = None
    predicted_value: Any = None
    predicted_score: float | None = None
    probability_by_class: dict[str, float] = Field(default_factory=dict)
    scored_row: dict[str, Any] = Field(default_factory=dict)


class BatchPredictionResult(BaseModel):
    """Normalized result for a batch prediction job."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    loaded_model: LoadedModel
    validation: PredictionValidationResult
    summary: PredictionSummary
    artifacts: PredictionArtifactBundle | None = None
    history_entry: PredictionHistoryEntry | None = None
    warnings: list[str] = Field(default_factory=list)
    scored_dataframe: pd.DataFrame
