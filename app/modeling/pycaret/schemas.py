"""Pydantic schemas for PyCaret experiment configuration and results."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config.enums import ExecutionBackend, WorkspaceMode


class ExperimentTaskType(str, Enum):
    """Supported PyCaret experiment task types."""

    AUTO = "auto"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ExperimentSortDirection(str, Enum):
    """Metric ordering direction."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


class MLflowTrackingMode(str, Enum):
    """How MLflow tracking should be handled for experiments."""

    OFF = "off"
    MANUAL = "manual"
    PYCARET_NATIVE = "pycaret_native"


class CustomMetricSpec(BaseModel):
    """Safe custom metric registration contract."""

    metric_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    target: Literal["pred", "pred_proba", "threshold"] = "pred"
    greater_is_better: bool = True
    kwargs: dict[str, Any] = Field(default_factory=dict)
    multiclass: bool = True

    @field_validator("metric_id", "display_name")
    @classmethod
    def _validate_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Custom metric fields must not be blank.")
        return value

    @field_validator("kwargs")
    @classmethod
    def _validate_safe_kwargs(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved = {
            "id",
            "name",
            "score_func",
            "target",
            "greater_is_better",
            "multiclass",
        }
        duplicate_keys = sorted(reserved.intersection(value))
        if duplicate_keys:
            raise ValueError(
                "Custom metric kwargs contain reserved names: "
                + ", ".join(duplicate_keys)
                + "."
            )

        for key, item in value.items():
            if not _is_json_safe_metric_kwarg(item):
                raise ValueError(
                    f"Custom metric kwarg '{key}' must be JSON-serializable and non-callable."
                )
        return value


class ExperimentSetupSummary(BaseModel):
    """Serializable record of the normalized setup stage."""

    normalized_config: dict[str, Any]
    actual_setup_kwargs: dict[str, Any]


class ExperimentSetupConfig(BaseModel):
    """Subset of setup options exposed by the experiment lab."""

    session_id: int | None = 42
    train_size: float = Field(default=0.7, gt=0.0, lt=1.0)
    fold: int = Field(default=5, ge=2)
    fold_strategy: str | None = None
    numeric_features: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)
    date_features: list[str] = Field(default_factory=list)
    ignore_features: list[str] = Field(default_factory=list)
    preprocess: bool = True
    experiment_name: str | None = None
    log_experiment: bool = False
    log_plots: bool | list[str] = False
    log_profile: bool = False
    log_data: bool = False
    system_log: bool = False
    n_jobs: int | None = -1
    use_gpu: bool | str = True
    html: bool = False

    def to_summary_model(self, *, actual_setup_kwargs: dict[str, Any]) -> ExperimentSetupSummary:
        """Return a serializable setup summary model."""

        return ExperimentSetupSummary(
            normalized_config=self.model_dump(mode="json"),
            actual_setup_kwargs=actual_setup_kwargs,
        )


class ExperimentCompareConfig(BaseModel):
    """Configuration for compare_models."""

    optimize: str | None = None
    n_select: int = Field(default=1, gt=0)
    include_models: list[str] = Field(default_factory=list)
    exclude_models: list[str] = Field(default_factory=list)
    turbo: bool = True
    budget_time: float | None = Field(default=None, gt=0)
    cross_validation: bool = True
    errors: str = "ignore"
    fit_kwargs: dict[str, Any] = Field(default_factory=dict)


class ExperimentTuneConfig(BaseModel):
    """Configuration for tune_model."""

    optimize: str | None = None
    n_iter: int = Field(default=10, gt=0)
    fold: int | None = Field(default=None, ge=2)
    custom_grid: dict[str, list[Any]] | None = None
    search_library: str = "scikit-learn"
    search_algorithm: str | None = None
    early_stopping: bool | str = False
    early_stopping_max_iters: int = Field(default=10, gt=0)
    choose_better: bool = True
    fit_kwargs: dict[str, Any] = Field(default_factory=dict)
    return_tuner: bool = False


class ExperimentEvaluationConfig(BaseModel):
    """Configuration for evaluation plots and interactive analysis."""

    plots: list[str] = Field(default_factory=list)
    plot_scale: float = Field(default=1.0, gt=0)
    plot_kwargs: dict[str, Any] = Field(default_factory=dict)
    interactive: bool = False


class ExperimentPersistenceConfig(BaseModel):
    """Configuration for model finalization and persistence."""

    save_experiment_snapshot: bool = True
    model_only: bool = False


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    target_column: str
    task_type: ExperimentTaskType = ExperimentTaskType.AUTO
    mlflow_tracking_mode: MLflowTrackingMode = MLflowTrackingMode.MANUAL
    custom_metrics: list[CustomMetricSpec] = Field(default_factory=list)
    setup: ExperimentSetupConfig = Field(default_factory=ExperimentSetupConfig)
    compare: ExperimentCompareConfig = Field(default_factory=ExperimentCompareConfig)
    tune: ExperimentTuneConfig = Field(default_factory=ExperimentTuneConfig)
    evaluation: ExperimentEvaluationConfig = Field(default_factory=ExperimentEvaluationConfig)
    persistence: ExperimentPersistenceConfig = Field(default_factory=ExperimentPersistenceConfig)


class ExperimentMetricRow(BaseModel):
    """Normalized metric row from get_metrics."""

    metric_id: str
    display_name: str
    greater_is_better: bool | None = None
    is_custom: bool = False
    raw_values: dict[str, Any] = Field(default_factory=dict)


class ModelSelectionSpec(BaseModel):
    """Stable user-facing model selection reference."""

    model_id: str | None = None
    model_name: str
    rank: int | None = None
    estimator_key: str | None = None


class ExperimentLeaderboardRow(BaseModel):
    """Normalized leaderboard row used across compare and tune flows."""

    stage: str = "compare"
    model_id: str | None = None
    model_name: str
    rank: int | None = None
    primary_score: float | None = None
    raw_metrics: dict[str, Any] = Field(default_factory=dict)
    estimator_key: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ExperimentTuneResult(BaseModel):
    """Summary of the baseline-to-tuned transition for one model."""

    selection: ModelSelectionSpec
    optimize_metric: str
    ranking_direction: ExperimentSortDirection
    applied_config: dict[str, Any] = Field(default_factory=dict)
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    tuned_metrics: dict[str, Any] = Field(default_factory=dict)
    baseline_score: float | None = None
    tuned_score: float | None = None
    tuner_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ExperimentPlotArtifact(BaseModel):
    """Metadata for one persisted evaluation plot."""

    plot_id: str
    model_name: str
    path: Path
    stage: str = "evaluate"
    plot_name: str | None = None
    warning: str | None = None


class SavedModelMetadata(BaseModel):
    """Stable saved-model metadata contract for future prediction flows."""

    task_type: ExperimentTaskType
    target_column: str
    model_id: str | None = None
    model_name: str
    model_path: Path
    dataset_name: str | None = None
    dataset_fingerprint: str | None = None
    trained_at: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    feature_dtypes: dict[str, str] = Field(default_factory=dict)
    target_dtype: str | None = None
    experiment_snapshot_path: Path | None = None
    experiment_snapshot_includes_data: bool = False
    model_only: bool = False
    artifact_format: str | None = None
    trusted_source: str | None = None
    model_sha256: str | None = None
    experiment_snapshot_sha256: str | None = None


class SavedModelArtifact(BaseModel):
    """Saved model metadata plus its persisted metadata sidecar path."""

    metadata: SavedModelMetadata
    metadata_path: Path | None = None


class ExperimentArtifactBundle(BaseModel):
    """Artifact paths emitted for an experiment run."""

    setup_json_path: Path | None = None
    metrics_csv_path: Path | None = None
    metrics_json_path: Path | None = None
    compare_csv_path: Path | None = None
    compare_json_path: Path | None = None
    tune_json_path: Path | None = None
    summary_json_path: Path | None = None
    markdown_summary_path: Path | None = None
    saved_model_metadata_path: Path | None = None
    saved_model_metadata_paths: list[Path] = Field(default_factory=list)
    experiment_snapshot_metadata_path: Path | None = None
    plot_artifacts: list[ExperimentPlotArtifact] = Field(default_factory=list)


class ExperimentSummary(BaseModel):
    """Roll-up summary for the current experiment state."""

    run_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_name: str | None = None
    dataset_fingerprint: str | None = None
    target_column: str
    task_type: ExperimentTaskType
    execution_backend: ExecutionBackend = ExecutionBackend.LOCAL
    workspace_mode: WorkspaceMode | None = None
    source_row_count: int
    source_column_count: int
    feature_column_count: int
    compare_optimize_metric: str | None = None
    compare_ranking_direction: ExperimentSortDirection | None = None
    tune_optimize_metric: str | None = None
    tune_ranking_direction: ExperimentSortDirection | None = None
    best_baseline_model_name: str | None = None
    best_baseline_score: float | None = None
    tuned_model_name: str | None = None
    tuned_score: float | None = None
    selected_model_name: str | None = None
    selected_model_id: str | None = None
    saved_model_name: str | None = None
    experiment_duration_seconds: float = 0.0
    setup_config: ExperimentSetupSummary
    warnings: list[str] = Field(default_factory=list)


class ExperimentRuntimeState(BaseModel):
    """Non-serializable runtime objects kept in memory only."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    experiment_handle: Any
    model_catalog: pd.DataFrame | None = None
    model_name_to_id: dict[str, str] = Field(default_factory=dict)
    created_models: dict[str, Any] = Field(default_factory=dict)
    tuned_models: dict[str, Any] = Field(default_factory=dict)
    compare_raw: pd.DataFrame | None = None


class ExperimentResultBundle(BaseModel):
    """Complete experiment package for UI, CLI, tracking, and persistence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_name: str | None = None
    dataset_fingerprint: str | None = None
    config: ExperimentConfig
    task_type: ExperimentTaskType
    execution_backend: ExecutionBackend = ExecutionBackend.LOCAL
    workspace_mode: WorkspaceMode | None = None
    feature_columns: list[str] = Field(default_factory=list)
    feature_dtypes: dict[str, str] = Field(default_factory=dict)
    target_dtype: str | None = None
    available_metrics: list[ExperimentMetricRow] = Field(default_factory=list)
    compare_leaderboard: list[ExperimentLeaderboardRow] = Field(default_factory=list)
    tuned_result: ExperimentTuneResult | None = None
    evaluation_plots: list[ExperimentPlotArtifact] = Field(default_factory=list)
    saved_model_metadata: SavedModelMetadata | None = None
    saved_model_artifacts: list[SavedModelArtifact] = Field(default_factory=list)
    artifacts: ExperimentArtifactBundle | None = None
    summary: ExperimentSummary
    warnings: list[str] = Field(default_factory=list)
    mlflow_run_id: str | None = None
    runtime: ExperimentRuntimeState | None = Field(default=None, exclude=True)


def _is_json_safe_metric_kwarg(value: Any) -> bool:
    if callable(value):
        return False
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe_metric_kwarg(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_safe_metric_kwarg(item) for key, item in value.items())
    return False