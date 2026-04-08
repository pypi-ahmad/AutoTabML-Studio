"""Pydantic schemas for FLAML AutoML configuration and results."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config.enums import ExecutionBackend, WorkspaceMode


class FlamlTaskType(str, Enum):
    """Supported FLAML task types."""

    AUTO = "auto"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class FlamlSortDirection(str, Enum):
    """Metric ordering direction."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


# ── Default estimator list ────────────────────────────────────────────
DEFAULT_ESTIMATOR_LIST: list[str] = [
    "lgbm",
    "xgboost",
    "xgb_limitdepth",
    "rf",
    "extra_tree",
    "lrl1",
    "lrl2",
    "kneighbor",
]

DEFAULT_CLASSIFICATION_METRIC = "accuracy"
DEFAULT_REGRESSION_METRIC = "r2"


# ── Configuration models ─────────────────────────────────────────────

class FlamlSearchConfig(BaseModel):
    """Configuration for the FLAML AutoML search."""

    time_budget: int = Field(default=120, gt=0)
    max_iter: int | None = Field(default=None, gt=0)
    metric: str = "auto"
    estimator_list: list[str] = Field(default_factory=lambda: list(DEFAULT_ESTIMATOR_LIST))
    eval_method: str = "auto"
    n_splits: int = Field(default=5, ge=2)
    split_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int | None = 0
    ensemble: bool = False
    early_stop: bool = False
    sample: bool = True
    n_jobs: int = -1
    verbose: int = 0
    retrain_full: bool = True
    model_history: bool = False
    log_training_metric: bool = False


class FlamlConfig(BaseModel):
    """Top-level FLAML AutoML configuration."""

    target_column: str
    task_type: FlamlTaskType = FlamlTaskType.AUTO
    search: FlamlSearchConfig = Field(default_factory=FlamlSearchConfig)


# ── Result models ────────────────────────────────────────────────────

class FlamlLeaderboardRow(BaseModel):
    """One row in the FLAML search history leaderboard."""

    rank: int | None = None
    estimator_name: str
    best_loss: float | None = None
    best_config: dict[str, Any] = Field(default_factory=dict)
    train_time: float | None = None


class FlamlSearchResult(BaseModel):
    """Summary of the FLAML AutoML search."""

    best_estimator: str | None = None
    best_config: dict[str, Any] = Field(default_factory=dict)
    best_loss: float | None = None
    best_config_train_time: float | None = None
    time_to_find_best: float | None = None
    metric: str | None = None
    leaderboard: list[FlamlLeaderboardRow] = Field(default_factory=list)


class FlamlSavedModelMetadata(BaseModel):
    """Metadata sidecar for a saved FLAML model."""

    task_type: FlamlTaskType
    target_column: str
    model_name: str
    model_path: Path
    best_estimator: str | None = None
    dataset_name: str | None = None
    dataset_fingerprint: str | None = None
    trained_at: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    feature_dtypes: dict[str, str] = Field(default_factory=dict)
    target_dtype: str | None = None
    best_config: dict[str, Any] = Field(default_factory=dict)
    best_loss: float | None = None
    metric: str | None = None
    search_duration_seconds: float = 0.0
    framework: str = "flaml"


class FlamlArtifactBundle(BaseModel):
    """Artifact paths emitted for a FLAML run."""

    search_result_json_path: Path | None = None
    leaderboard_csv_path: Path | None = None
    leaderboard_json_path: Path | None = None
    summary_json_path: Path | None = None
    saved_model_metadata_path: Path | None = None


class FlamlSummary(BaseModel):
    """Roll-up summary for the current FLAML run state."""

    run_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_name: str | None = None
    dataset_fingerprint: str | None = None
    target_column: str
    task_type: FlamlTaskType
    execution_backend: ExecutionBackend = ExecutionBackend.LOCAL
    workspace_mode: WorkspaceMode | None = None
    source_row_count: int
    source_column_count: int
    feature_column_count: int
    metric: str | None = None
    best_estimator: str | None = None
    best_loss: float | None = None
    best_config: dict[str, Any] = Field(default_factory=dict)
    search_duration_seconds: float = 0.0
    saved_model_name: str | None = None
    search_config: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class FlamlRuntimeState(BaseModel):
    """Non-serializable runtime objects kept in memory only."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    automl_instance: Any = Field(default=None, exclude=True)


class FlamlResultBundle(BaseModel):
    """Complete FLAML result package for UI, CLI, tracking, and persistence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_name: str | None = None
    dataset_fingerprint: str | None = None
    config: FlamlConfig
    task_type: FlamlTaskType
    execution_backend: ExecutionBackend = ExecutionBackend.LOCAL
    workspace_mode: WorkspaceMode | None = None
    feature_columns: list[str] = Field(default_factory=list)
    feature_dtypes: dict[str, str] = Field(default_factory=dict)
    target_dtype: str | None = None
    search_result: FlamlSearchResult | None = None
    saved_model_metadata: FlamlSavedModelMetadata | None = None
    artifacts: FlamlArtifactBundle | None = None
    summary: FlamlSummary
    warnings: list[str] = Field(default_factory=list)
    mlflow_run_id: str | None = None
    runtime: FlamlRuntimeState | None = Field(default=None, exclude=True)
