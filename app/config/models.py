"""Pydantic configuration models for AutoTabML Studio."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config.enums import (
    DEFAULT_MODELS,
    ExecutionBackend,
    LLMProvider,
    WorkspaceMode,
)

_ARTIFACT_ROOT = Path("artifacts")
_VALIDATION_DIR = _ARTIFACT_ROOT / "validation"
_PROFILING_DIR = _ARTIFACT_ROOT / "profiling"
_BENCHMARK_DIR = _ARTIFACT_ROOT / "benchmark"
_EXPERIMENTS_DIR = _ARTIFACT_ROOT / "experiments"
_MODELS_DIR = _ARTIFACT_ROOT / "models"
_SNAPSHOTS_DIR = _EXPERIMENTS_DIR / "snapshots"
_COMPARISONS_DIR = _ARTIFACT_ROOT / "comparisons"
_PREDICTIONS_DIR = _ARTIFACT_ROOT / "predictions"
_TEMP_DIR = _ARTIFACT_ROOT / "tmp"
_APP_DB_PATH = _ARTIFACT_ROOT / "app" / "app_metadata.sqlite3"
_PREDICTION_HISTORY_PATH = _PREDICTIONS_DIR / "history.jsonl"


class ArtifactSettings(BaseModel):
    """Canonical local artifact paths for workspace-generated files."""

    root_dir: Path = Field(default=_ARTIFACT_ROOT)
    validation_dir: Path = Field(default=_VALIDATION_DIR)
    profiling_dir: Path = Field(default=_PROFILING_DIR)
    benchmark_dir: Path = Field(default=_BENCHMARK_DIR)
    experiments_dir: Path = Field(default=_EXPERIMENTS_DIR)
    models_dir: Path = Field(default=_MODELS_DIR)
    snapshots_dir: Path = Field(default=_SNAPSHOTS_DIR)
    comparisons_dir: Path = Field(default=_COMPARISONS_DIR)
    predictions_dir: Path = Field(default=_PREDICTIONS_DIR)
    temp_dir: Path = Field(default=_TEMP_DIR)
    temp_retention_hours: int = Field(default=24, ge=1)
    failed_partial_retention_hours: int = Field(default=48, ge=1)


class DatabaseSettings(BaseModel):
    """Settings for the local app metadata database."""

    path: Path = Field(default=_APP_DB_PATH)
    initialize_on_startup: bool = True


class ProviderSettings(BaseModel):
    """Credentials and connection settings for a single LLM provider."""
    provider: LLMProvider = LLMProvider.OPENAI
    base_url: str | None = None

    @field_validator("base_url", mode="before")
    @classmethod
    def _normalize_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ExecutionSettings(BaseModel):
    """Settings related to the execution backend."""
    backend: ExecutionBackend = ExecutionBackend.COLAB_MCP


class UISettings(BaseModel):
    """Transient UI-level preferences."""
    selected_model_id: str | None = None


# ---------------------------------------------------------------------------
# Validation & Profiling configuration
# ---------------------------------------------------------------------------

class ValidationSeverity(str, Enum):
    """Severity levels for validation checks."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ProfilingMode(str, Enum):
    """Profiling report modes."""
    STANDARD = "standard"
    MINIMAL = "minimal"


class ValidationSettings(BaseModel):
    """Configuration for the data validation layer."""
    artifacts_dir: Path = Field(default=_VALIDATION_DIR)
    gx_context_dir: Path = Field(default=Path("gx"))
    data_docs_enabled: bool = False
    min_row_threshold: int = Field(default=1, ge=0)
    null_warn_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    null_fail_pct: float = Field(default=95.0, ge=0.0, le=100.0)


class ProfilingSettings(BaseModel):
    """Configuration for the automated profiling layer."""
    artifacts_dir: Path = Field(default=_PROFILING_DIR)
    default_mode: ProfilingMode = ProfilingMode.STANDARD
    large_dataset_row_threshold: int = Field(default=50_000, gt=0)
    large_dataset_col_threshold: int = Field(default=100, gt=0)
    sampling_row_threshold: int = Field(default=200_000, gt=0)
    sample_size: int = Field(default=50_000, gt=0)


class BenchmarkSettings(BaseModel):
    """Configuration for baseline model benchmarking."""

    artifacts_dir: Path = Field(default=_BENCHMARK_DIR)
    prefer_gpu: bool = True
    default_test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    default_random_state: int = 42
    default_stratify: bool = True
    default_classification_ranking_metric: str = "Balanced Accuracy"
    default_regression_ranking_metric: str = "Adjusted R-Squared"
    sampling_row_threshold: int = Field(default=100_000, gt=0)
    suggested_sample_rows: int = Field(default=50_000, gt=0)
    default_sample_rows: int | None = None
    timeout_seconds: float | None = Field(default=120.0, gt=0)
    mlflow_experiment_name: str = "autotabml-benchmarks"
    ui_default_top_k: int = Field(default=5, gt=0)


class PyCaretExperimentSettings(BaseModel):
    """Configuration for deeper PyCaret experiment workflows."""

    artifacts_dir: Path = Field(default=_EXPERIMENTS_DIR)
    models_dir: Path = Field(default=_MODELS_DIR)
    snapshots_dir: Path = Field(default=_SNAPSHOTS_DIR)
    default_session_id: int = 42
    default_train_size: float = Field(default=0.7, gt=0.0, lt=1.0)
    default_fold: int = Field(default=5, ge=2)
    default_classification_fold_strategy: str = "stratifiedkfold"
    default_regression_fold_strategy: str = "kfold"
    default_preprocess: bool = True
    default_compare_metric_classification: str = "Accuracy"
    default_compare_metric_regression: str = "R2"
    default_tune_metric_classification: str = "AUC"
    default_tune_metric_regression: str = "R2"
    default_plot_ids_classification: list[str] = Field(
        default_factory=lambda: [
            "confusion_matrix",
            "auc",
            "pr",
            "class_report",
            "feature",
        ]
    )
    default_plot_ids_regression: list[str] = Field(
        default_factory=lambda: [
            "residuals",
            "error",
            "feature",
        ]
    )
    default_use_gpu: bool | str = True
    default_tracking_mode: Literal["off", "manual", "pycaret_native"] = "manual"
    default_native_log_experiment: bool = False
    allow_log_plots: bool = False
    allow_log_profile: bool = False
    allow_log_data: bool = False
    mlflow_experiment_name: str = "autotabml-experiments"


class MLflowSettings(BaseModel):
    """Configuration for MLflow-backed tracking, comparison, and registry access."""

    tracking_uri: str | None = None
    registry_uri: str | None = None
    default_experiment_names: list[str] = Field(
        default_factory=lambda: ["autotabml-benchmarks", "autotabml-experiments"]
    )
    history_page_default_limit: int = Field(default=50, gt=0)
    champion_alias: str = "champion"
    candidate_alias: str = "candidate"
    archived_tag_key: str = "app.status"
    registry_enabled: bool = True
    comparison_artifacts_dir: Path = Field(default=_COMPARISONS_DIR)

    @field_validator("tracking_uri", "registry_uri", mode="before")
    @classmethod
    def _normalize_uris(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


TrackingSettings = MLflowSettings


class PredictionSettings(BaseModel):
    """Configuration for prediction / inference workflows."""

    artifacts_dir: Path = Field(default=_PREDICTIONS_DIR)
    default_output_stem: str = "predictions"
    history_path: Path = Field(default=_PREDICTION_HISTORY_PATH)
    schema_validation_mode: Literal["strict", "warn"] = "strict"
    prediction_column_name: str = "prediction"
    prediction_score_column_name: str = "prediction_score"
    supported_local_model_dirs: list[Path] = Field(default_factory=lambda: [_MODELS_DIR])
    local_model_metadata_dirs: list[Path] = Field(
        default_factory=lambda: [_EXPERIMENTS_DIR, _MODELS_DIR]
    )
    default_mlflow_run_artifact_path: str = "model"


class AppSettings(BaseModel):
    """Top-level application configuration with safe defaults."""
    workspace_mode: WorkspaceMode = WorkspaceMode.DASHBOARD
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    ui: UISettings = Field(default_factory=UISettings)
    artifacts: ArtifactSettings = Field(default_factory=ArtifactSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    profiling: ProfilingSettings = Field(default_factory=ProfilingSettings)
    benchmark: BenchmarkSettings = Field(default_factory=BenchmarkSettings)
    pycaret: PyCaretExperimentSettings = Field(default_factory=PyCaretExperimentSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    prediction: PredictionSettings = Field(default_factory=PredictionSettings)

    # --- Feature flags ---
    mlflow_descriptions_enabled: bool = True
    llm_descriptions_enabled: bool = False

    # --- Ollama-specific defaults ---
    ollama_base_url: str = "http://localhost:11434"

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_keys(cls, data):  # noqa: ANN001
        if not isinstance(data, dict):
            return data
        if "tracking" in data and "mlflow" not in data:
            data = dict(data)
            data["mlflow"] = data.pop("tracking")
        return data

    @model_validator(mode="after")
    def _synchronize_path_sections(self) -> AppSettings:
        root_relative_artifact_paths = [
            ("validation_dir", _VALIDATION_DIR),
            ("profiling_dir", _PROFILING_DIR),
            ("benchmark_dir", _BENCHMARK_DIR),
            ("experiments_dir", _EXPERIMENTS_DIR),
            ("models_dir", _MODELS_DIR),
            ("snapshots_dir", _SNAPSHOTS_DIR),
            ("comparisons_dir", _COMPARISONS_DIR),
            ("predictions_dir", _PREDICTIONS_DIR),
            ("temp_dir", _TEMP_DIR),
        ]
        if self.artifacts.root_dir != _ARTIFACT_ROOT:
            for target_name, default_value in root_relative_artifact_paths:
                if getattr(self.artifacts, target_name) == default_value:
                    setattr(
                        self.artifacts,
                        target_name,
                        self.artifacts.root_dir / default_value.relative_to(_ARTIFACT_ROOT),
                    )

        legacy_path_mappings = [
            (self.validation.artifacts_dir, _VALIDATION_DIR, "validation_dir"),
            (self.profiling.artifacts_dir, _PROFILING_DIR, "profiling_dir"),
            (self.benchmark.artifacts_dir, _BENCHMARK_DIR, "benchmark_dir"),
            (self.pycaret.artifacts_dir, _EXPERIMENTS_DIR, "experiments_dir"),
            (self.pycaret.models_dir, _MODELS_DIR, "models_dir"),
            (self.pycaret.snapshots_dir, _SNAPSHOTS_DIR, "snapshots_dir"),
            (self.mlflow.comparison_artifacts_dir, _COMPARISONS_DIR, "comparisons_dir"),
            (self.prediction.artifacts_dir, _PREDICTIONS_DIR, "predictions_dir"),
        ]
        for current_value, default_value, target_name in legacy_path_mappings:
            if getattr(self.artifacts, target_name) == default_value and current_value != default_value:
                setattr(self.artifacts, target_name, current_value)

        self.validation.artifacts_dir = self.artifacts.validation_dir
        self.profiling.artifacts_dir = self.artifacts.profiling_dir
        self.benchmark.artifacts_dir = self.artifacts.benchmark_dir
        self.pycaret.artifacts_dir = self.artifacts.experiments_dir
        self.pycaret.models_dir = self.artifacts.models_dir
        self.pycaret.snapshots_dir = self.artifacts.snapshots_dir
        self.mlflow.comparison_artifacts_dir = self.artifacts.comparisons_dir
        self.prediction.artifacts_dir = self.artifacts.predictions_dir

        if self.database.path == _APP_DB_PATH and self.artifacts.root_dir != _ARTIFACT_ROOT:
            self.database.path = self.artifacts.root_dir / "app" / _APP_DB_PATH.name

        if self.prediction.history_path == _PREDICTION_HISTORY_PATH:
            self.prediction.history_path = self.artifacts.predictions_dir / _PREDICTION_HISTORY_PATH.name

        if self.prediction.supported_local_model_dirs == [_MODELS_DIR]:
            self.prediction.supported_local_model_dirs = [self.artifacts.models_dir]

        if self.prediction.local_model_metadata_dirs == [_EXPERIMENTS_DIR, _MODELS_DIR]:
            self.prediction.local_model_metadata_dirs = [
                self.artifacts.experiments_dir,
                self.artifacts.models_dir,
            ]

        self.ollama_base_url = self.ollama_base_url.strip().rstrip("/") or "http://localhost:11434"
        return self

    @property
    def tracking(self) -> MLflowSettings:
        """Backward-compatible alias for the canonical MLflow settings section."""

        return self.mlflow

    @tracking.setter
    def tracking(self, value: MLflowSettings) -> None:
        self.mlflow = value

    def default_model_for_provider(self, provider: LLMProvider | None = None) -> str | None:
        """Return the verified stable fallback model id for the given (or current) provider."""
        p = provider or self.provider.provider
        return DEFAULT_MODELS.get(p)
