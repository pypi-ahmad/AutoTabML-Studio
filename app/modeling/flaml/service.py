"""FLAML AutoML service — search, save, and track."""

from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from app.config.enums import ExecutionBackend, WorkspaceMode
from app.errors import log_and_wrap, log_exception
from app.modeling.base import BaseService
from app.modeling.flaml.artifacts import write_flaml_artifacts
from app.modeling.flaml.errors import (
    FlamlDependencyError,
    FlamlExecutionError,
    FlamlTargetError,
)
from app.modeling.flaml.mlflow_tracking import MLflowFlamlTracker
from app.modeling.flaml.schemas import (
    DEFAULT_CLASSIFICATION_METRIC,
    DEFAULT_REGRESSION_METRIC,
    FlamlArtifactBundle,
    FlamlConfig,
    FlamlLeaderboardRow,
    FlamlResultBundle,
    FlamlRuntimeState,
    FlamlSavedModelMetadata,
    FlamlSearchResult,
    FlamlSortDirection,
    FlamlSummary,
    FlamlTaskType,
)
from app.modeling.flaml.setup_runner import require_flaml, resolve_task_type
from app.path_utils import safe_artifact_stem
from app.security.trusted_artifacts import TRUSTED_MODEL_SOURCE, compute_sha256, write_checksum_file
from app.storage import AppMetadataStore

logger = logging.getLogger(__name__)

LOWER_IS_BETTER_MARKERS = ("rmse", "mae", "mse", "rmsle", "mape", "loss", "error")


def metric_sort_direction(metric_name: str) -> FlamlSortDirection:
    """Return the expected ordering direction for a metric name."""

    lowered = metric_name.lower()
    if any(marker in lowered for marker in LOWER_IS_BETTER_MARKERS):
        return FlamlSortDirection.ASCENDING
    return FlamlSortDirection.DESCENDING


class FlamlAutoMLService(BaseService):
    """FLAML-backed AutoML service."""

    def __init__(
        self,
        *,
        artifacts_dir: Path | None = None,
        models_dir: Path | None = None,
        default_classification_metric: str = DEFAULT_CLASSIFICATION_METRIC,
        default_regression_metric: str = DEFAULT_REGRESSION_METRIC,
        mlflow_experiment_name: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        metadata_store: AppMetadataStore | None = None,
    ) -> None:
        super().__init__(
            artifacts_dir=artifacts_dir,
            mlflow_experiment_name=mlflow_experiment_name,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            metadata_store=metadata_store,
        )
        self._models_dir = models_dir
        self._default_classification_metric = default_classification_metric
        self._default_regression_metric = default_regression_metric

    def run_automl(
        self,
        df: pd.DataFrame,
        config: FlamlConfig,
        *,
        dataset_name: str | None = None,
        dataset_fingerprint: str | None = None,
        execution_backend: ExecutionBackend = ExecutionBackend.LOCAL,
        workspace_mode: WorkspaceMode | None = None,
    ) -> FlamlResultBundle:
        """Run a complete FLAML AutoML search."""

        if execution_backend != ExecutionBackend.LOCAL:
            raise FlamlExecutionError(
                f"FLAML execution for backend '{execution_backend.value}' is not implemented yet."
            )

        require_flaml()

        if config.target_column not in df.columns:
            raise FlamlTargetError(
                f"Target column '{config.target_column}' was not found in the dataset."
            )

        started_at = perf_counter()
        working_df = df.copy()
        warnings: list[str] = []

        # Drop null targets
        dropped_null_targets = int(working_df[config.target_column].isna().sum())
        if dropped_null_targets > 0:
            working_df = working_df.loc[working_df[config.target_column].notna()].copy()
            warnings.append(
                f"Dropped {dropped_null_targets} row(s) with null target values before search."
            )

        if working_df.empty:
            raise FlamlTargetError("No rows remain after dropping null target values.")

        # Resolve task type
        task_type, task_warnings = resolve_task_type(
            working_df[config.target_column],
            config.task_type,
        )
        warnings.extend(task_warnings)

        # Separate features and target
        feature_columns = [col for col in working_df.columns if col != config.target_column]
        feature_dtypes = {col: str(working_df[col].dtype) for col in feature_columns}
        target_dtype = str(working_df[config.target_column].dtype)

        X = working_df[feature_columns]
        y = working_df[config.target_column]

        # Resolve metric
        search_config = config.search
        metric = search_config.metric
        if metric == "auto":
            metric = (
                self._default_classification_metric
                if task_type == FlamlTaskType.CLASSIFICATION
                else self._default_regression_metric
            )

        # Run FLAML
        from flaml import AutoML

        automl = AutoML()
        fit_kwargs: dict[str, Any] = {
            "X_train": X,
            "y_train": y,
            "task": task_type.value,
            "metric": metric,
            "time_budget": search_config.time_budget,
            "estimator_list": list(search_config.estimator_list),
            "eval_method": search_config.eval_method,
            "n_splits": search_config.n_splits,
            "split_ratio": search_config.split_ratio,
            "seed": search_config.seed,
            "ensemble": search_config.ensemble,
            "early_stop": search_config.early_stop,
            "sample": search_config.sample,
            "n_jobs": search_config.n_jobs,
            "verbose": search_config.verbose,
            "retrain_full": search_config.retrain_full,
            "model_history": search_config.model_history,
            "log_training_metric": search_config.log_training_metric,
            "log_file_name": "",
        }
        if search_config.max_iter is not None:
            fit_kwargs["max_iter"] = search_config.max_iter

        try:
            automl.fit(**fit_kwargs)
        except ImportError as exc:
            log_and_wrap(
                logger,
                exc,
                operation="flaml.run_automl",
                wrap_with=FlamlDependencyError,
                message=str(exc),
                context={"task_type": task_type.value},
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            log_and_wrap(
                logger,
                exc,
                operation="flaml.run_automl",
                wrap_with=FlamlExecutionError,
                message=f"FLAML search failed: {exc}",
                context={"task_type": task_type.value},
            )

        search_duration = round(perf_counter() - started_at, 4)

        # Build leaderboard from search history
        leaderboard = self._build_leaderboard(automl)

        search_result = FlamlSearchResult(
            best_estimator=automl.best_estimator,
            best_config=dict(automl.best_config) if automl.best_config else {},
            best_loss=automl.best_loss,
            best_config_train_time=automl.best_config_train_time,
            time_to_find_best=automl.time_to_find_best_model,
            metric=metric,
            leaderboard=leaderboard,
        )

        runtime = FlamlRuntimeState(automl_instance=automl)

        summary = FlamlSummary(
            dataset_name=dataset_name,
            dataset_fingerprint=dataset_fingerprint,
            target_column=config.target_column,
            task_type=task_type,
            execution_backend=execution_backend,
            workspace_mode=workspace_mode,
            source_row_count=len(working_df),
            source_column_count=len(working_df.columns),
            feature_column_count=len(feature_columns),
            metric=metric,
            best_estimator=automl.best_estimator,
            best_loss=automl.best_loss,
            best_config=dict(automl.best_config) if automl.best_config else {},
            search_duration_seconds=search_duration,
            search_config=search_config.model_dump(mode="json"),
            warnings=list(warnings),
        )

        bundle = FlamlResultBundle(
            dataset_name=dataset_name,
            dataset_fingerprint=dataset_fingerprint,
            config=config,
            task_type=task_type,
            execution_backend=execution_backend,
            workspace_mode=workspace_mode,
            feature_columns=feature_columns,
            feature_dtypes=feature_dtypes,
            target_dtype=target_dtype,
            search_result=search_result,
            summary=summary,
            warnings=list(warnings),
            runtime=runtime,
        )

        self._refresh_artifacts(bundle)
        self._track_bundle(bundle)
        return bundle

    def save_best_model(
        self,
        bundle: FlamlResultBundle,
        *,
        save_name: str,
    ) -> FlamlResultBundle:
        """Persist the best model from a FLAML search."""

        runtime = bundle.runtime
        if runtime is None or runtime.automl_instance is None:
            raise FlamlExecutionError("No FLAML runtime available — run search first.")

        models_dir = self._resolve_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)

        base_name = safe_artifact_stem(save_name)
        model_path = models_dir / f"{base_name}.pkl"
        with model_path.open("wb") as f:
            pickle.dump(bundle.runtime.automl_instance, f)
        model_sha256 = compute_sha256(model_path)
        write_checksum_file(model_path, checksum=model_sha256)

        metadata = FlamlSavedModelMetadata(
            task_type=bundle.task_type,
            target_column=bundle.config.target_column,
            model_name=save_name,
            model_path=model_path,
            best_estimator=bundle.search_result.best_estimator if bundle.search_result else None,
            dataset_name=bundle.dataset_name,
            dataset_fingerprint=bundle.dataset_fingerprint,
            trained_at=datetime.now(timezone.utc).isoformat(),
            feature_columns=bundle.feature_columns,
            feature_dtypes=bundle.feature_dtypes,
            target_dtype=bundle.target_dtype,
            best_config=bundle.search_result.best_config if bundle.search_result else {},
            best_loss=bundle.search_result.best_loss if bundle.search_result else None,
            metric=bundle.search_result.metric if bundle.search_result else None,
            search_duration_seconds=bundle.summary.search_duration_seconds,
            artifact_format="flaml_pickle",
            trusted_source=TRUSTED_MODEL_SOURCE,
            model_sha256=model_sha256,
        )

        # Write metadata sidecar
        metadata_path = models_dir / f"{base_name}_flaml_saved_model_metadata_{base_name}.json"
        metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        write_checksum_file(metadata_path)

        bundle.saved_model_metadata = metadata
        if bundle.artifacts is None:
            bundle.artifacts = FlamlArtifactBundle()
        bundle.artifacts.saved_model_metadata_path = metadata_path
        bundle.summary.saved_model_name = save_name

        self._refresh_artifacts(bundle)
        self._track_bundle(bundle)
        return bundle

    def _build_leaderboard(self, automl: Any) -> list[FlamlLeaderboardRow]:
        """Build a leaderboard from FLAML's best_loss_per_estimator."""

        rows: list[FlamlLeaderboardRow] = []
        try:
            best_loss_per = automl.best_loss_per_estimator
            best_config_per = automl.best_config_per_estimator
        except (AttributeError, TypeError, ValueError) as exc:
            log_exception(logger, exc, operation="flaml.build_leaderboard", level=logging.DEBUG)
            return rows

        if not best_loss_per:
            return rows

        sorted_estimators = sorted(best_loss_per.items(), key=lambda x: x[1])
        for rank, (estimator_name, loss) in enumerate(sorted_estimators, start=1):
            config = best_config_per.get(estimator_name, {}) if best_config_per else {}
            rows.append(
                FlamlLeaderboardRow(
                    rank=rank,
                    estimator_name=estimator_name,
                    best_loss=loss,
                    best_config=dict(config) if config else {},
                )
            )
        return rows

    def _refresh_artifacts(self, bundle: FlamlResultBundle) -> None:
        artifacts_dir = self._resolve_artifacts_dir()
        if artifacts_dir is None:
            return
        try:
            bundle.artifacts = write_flaml_artifacts(bundle, artifacts_dir)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            log_exception(logger, exc, operation="flaml.write_artifacts")
            bundle.warnings.append(f"Artifact writing failed: {exc}")

    def _track_bundle(self, bundle: FlamlResultBundle) -> None:
        tracker = self._build_tracker(MLflowFlamlTracker)
        if tracker is None:
            return
        run_id, warnings = tracker.log_bundle(
            bundle,
            existing_run_id=bundle.mlflow_run_id,
        )
        bundle.mlflow_run_id = run_id
        self._append_bundle_warnings(bundle, warnings)

    def _resolve_artifacts_dir(self) -> Path | None:
        return self._artifacts_dir

    def _resolve_models_dir(self) -> Path:
        if self._models_dir is not None:
            return self._models_dir
        from app.config.models import _MODELS_DIR

        return _MODELS_DIR
