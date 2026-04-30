"""Production-style service wrapper around the PyCaret experiment OOP API."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import pandas as pd

from app.config.enums import ExecutionBackend, WorkspaceMode
from app.errors import log_and_wrap, log_exception
from app.modeling.pycaret import setup_runner
from app.modeling.pycaret.artifacts import write_experiment_artifacts
from app.modeling.pycaret.base import BaseExperimentService
from app.modeling.pycaret.compare_runner import create_model, run_compare_models
from app.modeling.pycaret.errors import (
    PyCaretDependencyError,
    PyCaretExecutionError,
    PyCaretTargetError,
)
from app.modeling.pycaret.evaluate_runner import generate_evaluation_plots
from app.modeling.pycaret.metrics_runner import add_custom_metric, list_available_metrics, remove_custom_metric
from app.modeling.pycaret.mlflow_tracking import MLflowExperimentTracker
from app.modeling.pycaret.persistence import save_finalized_model, write_saved_model_metadata_sidecar
from app.modeling.pycaret.schemas import (
    CustomMetricSpec,
    ExperimentConfig,
    ExperimentResultBundle,
    ExperimentRuntimeState,
    ExperimentSummary,
    ExperimentTaskType,
    ExperimentTuneResult,
    MLflowTrackingMode,
    ModelSelectionSpec,
    SavedModelArtifact,
)
from app.modeling.pycaret.selectors import metric_sort_direction, resolve_model_id, resolve_task_type
from app.modeling.pycaret.summary import extract_mean_metrics, normalize_compare_grid
from app.path_utils import model_save_name
from app.storage import AppMetadataStore, record_experiment_job

logger = logging.getLogger(__name__)


class PyCaretExperimentService(BaseExperimentService):
    """PyCaret-backed experiment service using the OOP API."""

    def __init__(
        self,
        *,
        artifacts_dir: Path | None,
        models_dir: Path | None,
        snapshots_dir: Path | None,
        classification_compare_metric: str,
        regression_compare_metric: str,
        classification_tune_metric: str,
        regression_tune_metric: str,
        mlflow_experiment_name: str | None,
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
        self._snapshots_dir = snapshots_dir
        self._classification_compare_metric = classification_compare_metric
        self._regression_compare_metric = regression_compare_metric
        self._classification_tune_metric = classification_tune_metric
        self._regression_tune_metric = regression_tune_metric

    def setup_experiment(
        self,
        df: pd.DataFrame,
        config: ExperimentConfig,
        *,
        test_df: pd.DataFrame | None = None,
        dataset_name: str | None = None,
        dataset_fingerprint: str | None = None,
        execution_backend: ExecutionBackend = ExecutionBackend.LOCAL,
        workspace_mode: WorkspaceMode | None = None,
    ) -> ExperimentResultBundle:
        if execution_backend != ExecutionBackend.LOCAL:
            raise PyCaretExecutionError(
                f"Experiment execution for backend '{execution_backend.value}' is not implemented yet. "
                "TODO: add remote PyCaret experiment execution scaffolding for this backend."
            )

        setup_runner.require_pycaret()
        if config.target_column not in df.columns:
            raise PyCaretTargetError(
                f"Target column '{config.target_column}' was not found in the dataset."
            )

        started_at = perf_counter()
        working_df = df.copy()
        dropped_null_targets = int(working_df[config.target_column].isna().sum())
        warnings: list[str] = []
        if dropped_null_targets > 0:
            working_df = working_df.loc[working_df[config.target_column].notna()].copy()
            warnings.append(
                f"Dropped {dropped_null_targets} row(s) with null target values before setup."
            )

        task_type, task_warnings = resolve_task_type(
            working_df[config.target_column],
            config.task_type,
        )
        warnings.extend(task_warnings)

        experiment_handle = setup_runner.build_pycaret_experiment(task_type)
        setup_kwargs, actual_setup_kwargs = setup_runner.build_setup_call_kwargs(
            config,
            task_type=task_type,
            mlflow_experiment_name=self._mlflow_experiment_name,
            test_data_supplied=test_df is not None,
        )

        try:
            experiment_handle.setup(
                working_df,
                test_data=test_df,
                **setup_kwargs,
            )
        except ImportError as exc:
            log_and_wrap(
                logger,
                exc,
                operation="pycaret.setup_experiment",
                wrap_with=PyCaretDependencyError,
                message=str(exc),
                context={"task_type": task_type.value},
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - exercised through tests
            log_and_wrap(
                logger,
                exc,
                operation="pycaret.setup_experiment",
                wrap_with=PyCaretExecutionError,
                message=f"PyCaret setup failed: {exc}",
                context={"task_type": task_type.value},
            )

        available_metrics = list_available_metrics(experiment_handle)
        model_catalog = experiment_handle.models(internal=False, raise_errors=False)
        model_name_to_id = {
            str(row["Name"]): str(index)
            for index, row in model_catalog.iterrows()
            if "Name" in row and row["Name"] is not None
        }

        feature_columns = [column for column in working_df.columns if column != config.target_column]
        feature_dtypes = {column: str(working_df[column].dtype) for column in feature_columns}
        target_dtype = str(working_df[config.target_column].dtype)

        runtime = ExperimentRuntimeState(
            experiment_handle=experiment_handle,
            model_catalog=model_catalog,
            model_name_to_id=model_name_to_id,
        )
        summary = ExperimentSummary(
            dataset_name=dataset_name,
            dataset_fingerprint=dataset_fingerprint,
            target_column=config.target_column,
            task_type=task_type,
            execution_backend=execution_backend,
            workspace_mode=workspace_mode,
            source_row_count=len(working_df),
            source_column_count=len(working_df.columns),
            feature_column_count=len(feature_columns),
            experiment_duration_seconds=round(perf_counter() - started_at, 4),
            setup_config=config.setup.to_summary_model(actual_setup_kwargs=actual_setup_kwargs),
            warnings=list(warnings),
        )
        bundle = ExperimentResultBundle(
            dataset_name=dataset_name,
            dataset_fingerprint=dataset_fingerprint,
            config=config,
            task_type=task_type,
            execution_backend=execution_backend,
            workspace_mode=workspace_mode,
            feature_columns=feature_columns,
            feature_dtypes=feature_dtypes,
            target_dtype=target_dtype,
            available_metrics=available_metrics,
            summary=summary,
            warnings=list(warnings),
            runtime=runtime,
        )
        self._refresh_artifacts(bundle)
        self._track_bundle(bundle)
        return bundle

    def run_compare_pipeline(
        self,
        df: pd.DataFrame,
        config: ExperimentConfig,
        **kwargs,
    ) -> ExperimentResultBundle:
        """Convenience setup + compare pipeline for CLI use."""

        bundle = self.setup_experiment(df, config, **kwargs)
        return self.compare_models(bundle)

    def compare_models(self, bundle: ExperimentResultBundle) -> ExperimentResultBundle:
        runtime = self._require_runtime(bundle)
        optimize_metric = bundle.config.compare.optimize or self._default_compare_metric(bundle.task_type)
        started_at = perf_counter()
        run_compare_models(runtime.experiment_handle, bundle.config.compare, optimize_metric=optimize_metric)
        score_grid = runtime.experiment_handle.pull()
        runtime.compare_raw = score_grid.copy()

        rows, resolved_metric, direction, warnings = normalize_compare_grid(
            score_grid,
            requested_metric=optimize_metric,
            model_name_to_id=runtime.model_name_to_id,
        )
        bundle.compare_leaderboard = rows
        bundle.summary.compare_optimize_metric = resolved_metric
        bundle.summary.compare_ranking_direction = direction
        if rows:
            bundle.summary.best_baseline_model_name = rows[0].model_name
            bundle.summary.best_baseline_score = rows[0].primary_score
        bundle.summary.experiment_duration_seconds = round(
            bundle.summary.experiment_duration_seconds + (perf_counter() - started_at),
            4,
        )
        self._append_warnings(bundle, warnings)
        self._refresh_artifacts(bundle)
        self._track_bundle(bundle)
        return bundle

    def add_custom_metric(
        self,
        bundle: ExperimentResultBundle,
        spec: CustomMetricSpec,
        *,
        score_func,
    ) -> ExperimentResultBundle:
        runtime = self._require_runtime(bundle)
        add_custom_metric(
            runtime.experiment_handle,
            task_type=bundle.task_type,
            spec=spec,
            score_func=score_func,
        )
        bundle.available_metrics = list_available_metrics(runtime.experiment_handle)
        bundle.config.custom_metrics.append(spec)
        self._refresh_artifacts(bundle)
        self._track_bundle(bundle)
        return bundle

    def remove_custom_metric(self, bundle: ExperimentResultBundle, name_or_id: str) -> ExperimentResultBundle:
        runtime = self._require_runtime(bundle)
        remove_custom_metric(runtime.experiment_handle, name_or_id)
        bundle.available_metrics = list_available_metrics(runtime.experiment_handle)
        bundle.config.custom_metrics = [
            spec
            for spec in bundle.config.custom_metrics
            if spec.metric_id != name_or_id and spec.display_name != name_or_id
        ]
        self._refresh_artifacts(bundle)
        self._track_bundle(bundle)
        return bundle

    def tune_model(
        self,
        bundle: ExperimentResultBundle,
        selection: ModelSelectionSpec,
    ) -> ExperimentResultBundle:
        from app.modeling.pycaret.tune_runner import run_tune_model

        runtime = self._require_runtime(bundle)
        started_at = perf_counter()
        baseline_model, baseline_metrics, estimator_key = self._create_model_for_selection(bundle, selection)
        optimize_metric = bundle.config.tune.optimize or self._default_tune_metric(bundle.task_type)
        tuned_model, tuner_object = run_tune_model(
            runtime.experiment_handle,
            baseline_model,
            bundle.config.tune,
            optimize_metric=optimize_metric,
        )
        tuned_metrics = extract_mean_metrics(runtime.experiment_handle.pull())
        resolved_model_id = resolve_model_id(selection, runtime.model_name_to_id)
        runtime.tuned_models[f"tuned::{resolved_model_id or selection.model_name}"] = tuned_model

        direction = metric_sort_direction(optimize_metric)
        tuned_result = ExperimentTuneResult(
            selection=ModelSelectionSpec(
                model_id=resolved_model_id,
                model_name=selection.model_name,
                rank=selection.rank,
                estimator_key=estimator_key,
            ),
            optimize_metric=optimize_metric,
            ranking_direction=direction,
            applied_config={
                **bundle.config.tune.model_dump(mode="json"),
                "optimize": optimize_metric,
            },
            baseline_metrics=baseline_metrics,
            tuned_metrics=tuned_metrics,
            baseline_score=self._metric_value(baseline_metrics, optimize_metric),
            tuned_score=self._metric_value(tuned_metrics, optimize_metric),
            tuner_summary={"tuner_object": str(type(tuner_object).__name__) if tuner_object is not None else ""},
        )
        bundle.tuned_result = tuned_result
        bundle.summary.selected_model_id = resolved_model_id
        bundle.summary.selected_model_name = selection.model_name
        bundle.summary.tune_optimize_metric = optimize_metric
        bundle.summary.tune_ranking_direction = direction
        bundle.summary.tuned_model_name = selection.model_name
        bundle.summary.tuned_score = tuned_result.tuned_score
        bundle.summary.experiment_duration_seconds = round(
            bundle.summary.experiment_duration_seconds + (perf_counter() - started_at),
            4,
        )
        self._refresh_artifacts(bundle)
        self._track_bundle(bundle)
        return bundle

    def evaluate_model(
        self,
        bundle: ExperimentResultBundle,
        selection: ModelSelectionSpec,
    ) -> ExperimentResultBundle:
        runtime = self._require_runtime(bundle)
        model, model_name = self._resolve_model_for_action(bundle, selection)
        bundle.summary.selected_model_id = resolve_model_id(selection, runtime.model_name_to_id)
        bundle.summary.selected_model_name = model_name
        artifacts_dir = self._resolve_artifacts_dir()
        plot_artifacts, warnings = generate_evaluation_plots(
            runtime.experiment_handle,
            model,
            task_type=bundle.task_type,
            model_name=model_name,
            evaluation=bundle.config.evaluation,
            output_dir=artifacts_dir,
        )
        bundle.evaluation_plots = plot_artifacts
        self._append_warnings(bundle, warnings)
        self._refresh_artifacts(bundle)
        self._track_bundle(bundle)
        return bundle

    def finalize_and_save_model(
        self,
        bundle: ExperimentResultBundle,
        selection: ModelSelectionSpec,
        *,
        save_name: str,
    ) -> ExperimentResultBundle:
        runtime = self._require_runtime(bundle)
        resolved_model_id = resolve_model_id(selection, runtime.model_name_to_id)
        saved_artifact = self._save_selection_model(
            bundle,
            selection,
            save_name=save_name,
            save_experiment_snapshot=bundle.config.persistence.save_experiment_snapshot,
        )
        bundle.saved_model_metadata = saved_artifact.metadata
        bundle.saved_model_artifacts = self._upsert_saved_model_artifact(
            bundle.saved_model_artifacts,
            saved_artifact,
        )
        bundle.summary.selected_model_id = resolved_model_id
        bundle.summary.selected_model_name = selection.model_name
        bundle.summary.saved_model_name = saved_artifact.metadata.model_name
        self._refresh_artifacts(bundle)
        self._track_bundle(bundle)
        return bundle

    def finalize_and_save_all_models(
        self,
        bundle: ExperimentResultBundle,
        *,
        save_name_prefix: str | None = None,
        include_experiment_snapshots: bool = False,
        max_models: int | None = None,
    ) -> ExperimentResultBundle:
        self._require_runtime(bundle)
        if not bundle.compare_leaderboard:
            raise PyCaretExecutionError(
                "No compared models are available to save. Run compare_models first."
            )

        rows_to_save = list(bundle.compare_leaderboard)
        if max_models is not None and max_models > 0:
            rows_to_save = rows_to_save[:max_models]

        saved_count = 0
        warnings: list[str] = []

        for row in rows_to_save:
            selection = ModelSelectionSpec(
                model_id=row.model_id,
                model_name=row.model_name,
                rank=row.rank,
            )
            save_name = self._build_bulk_save_name(
                bundle,
                selection,
                save_name_prefix=save_name_prefix,
            )
            try:
                saved_artifact = self._save_selection_model(
                    bundle,
                    selection,
                    save_name=save_name,
                    save_experiment_snapshot=include_experiment_snapshots,
                )
            except (PyCaretDependencyError, PyCaretExecutionError, OSError, RuntimeError, TypeError, ValueError) as exc:
                log_exception(
                    logger,
                    exc,
                    operation="pycaret.save_selection_model",
                    context={"model_name": selection.model_name, "save_name": save_name},
                )
                warnings.append(f"Could not save model '{selection.model_name}': {exc}")
                continue

            saved_count += 1
            if bundle.saved_model_metadata is None:
                bundle.saved_model_metadata = saved_artifact.metadata
            bundle.saved_model_artifacts = self._upsert_saved_model_artifact(
                bundle.saved_model_artifacts,
                saved_artifact,
            )

        if saved_count == 0:
            raise PyCaretExecutionError(
                "Automatic save could not persist any compared model."
            )

        top_row = bundle.compare_leaderboard[0]
        bundle.summary.selected_model_id = top_row.model_id
        bundle.summary.selected_model_name = top_row.model_name
        bundle.summary.saved_model_name = f"{saved_count} model(s) saved for prediction"
        self._append_warnings(bundle, warnings)
        self._refresh_artifacts(bundle)
        self._track_bundle(bundle)
        return bundle

    def _create_model_for_selection(
        self,
        bundle: ExperimentResultBundle,
        selection: ModelSelectionSpec,
    ) -> tuple[object, dict[str, object], str]:
        runtime = self._require_runtime(bundle)
        resolved_model_id = resolve_model_id(selection, runtime.model_name_to_id)
        if resolved_model_id is None:
            raise PyCaretExecutionError(
                f"Could not resolve a PyCaret model id for '{selection.model_name}'."
            )

        estimator_key = f"baseline::{resolved_model_id}"
        if estimator_key in runtime.created_models:
            model = runtime.created_models[estimator_key]
        else:
            model = create_model(
                runtime.experiment_handle,
                resolved_model_id,
                fit_kwargs=bundle.config.compare.fit_kwargs or None,
            )
            runtime.created_models[estimator_key] = model
        baseline_metrics = extract_mean_metrics(runtime.experiment_handle.pull())
        return model, baseline_metrics, estimator_key

    def _resolve_model_for_action(
        self,
        bundle: ExperimentResultBundle,
        selection: ModelSelectionSpec,
    ) -> tuple[object, str]:
        runtime = self._require_runtime(bundle)
        resolved_model_id = resolve_model_id(selection, runtime.model_name_to_id)
        tuned_key = f"tuned::{resolved_model_id or selection.model_name}"
        if tuned_key in runtime.tuned_models:
            return runtime.tuned_models[tuned_key], selection.model_name

        baseline_key = f"baseline::{resolved_model_id}"
        if baseline_key in runtime.created_models:
            return runtime.created_models[baseline_key], selection.model_name

        model, _, _ = self._create_model_for_selection(bundle, selection)
        return model, selection.model_name

    def _save_selection_model(
        self,
        bundle: ExperimentResultBundle,
        selection: ModelSelectionSpec,
        *,
        save_name: str,
        save_experiment_snapshot: bool,
    ) -> SavedModelArtifact:
        runtime = self._require_runtime(bundle)
        model, model_name = self._resolve_model_for_action(bundle, selection)
        resolved_model_id = resolve_model_id(selection, runtime.model_name_to_id)

        finalized_model = runtime.experiment_handle.finalize_model(
            model,
            model_only=bundle.config.persistence.model_only,
        )
        saved_metadata = save_finalized_model(
            runtime.experiment_handle,
            finalized_model,
            task_type=bundle.task_type,
            target_column=bundle.config.target_column,
            model_id=resolved_model_id,
            model_name=save_name,
            save_name=save_name,
            models_dir=self._resolve_models_dir(),
            snapshots_dir=self._resolve_snapshots_dir(),
            dataset_name=bundle.dataset_name,
            dataset_fingerprint=bundle.dataset_fingerprint,
            feature_columns=bundle.feature_columns,
            feature_dtypes=bundle.feature_dtypes,
            target_dtype=bundle.target_dtype,
            save_experiment_snapshot=save_experiment_snapshot,
            model_only=bundle.config.persistence.model_only,
        )
        metadata_path = write_saved_model_metadata_sidecar(
            saved_metadata,
            output_dir=self._resolve_models_dir(),
            stem=save_name,
            timestamp=bundle.summary.run_timestamp,
        )

        if self._metadata_store is not None:
            self._metadata_store.upsert_saved_model_metadata(saved_metadata, metadata_path=metadata_path)

        return SavedModelArtifact(metadata=saved_metadata, metadata_path=metadata_path)

    def _build_bulk_save_name(
        self,
        bundle: ExperimentResultBundle,
        selection: ModelSelectionSpec,
        *,
        save_name_prefix: str | None = None,
    ) -> str:
        return model_save_name(bundle.dataset_name, selection.model_name or selection.model_id)

    def _upsert_saved_model_artifact(
        self,
        artifacts: list[SavedModelArtifact],
        saved_artifact: SavedModelArtifact,
    ) -> list[SavedModelArtifact]:
        current_path = str(saved_artifact.metadata.model_path)
        retained = [item for item in artifacts if str(item.metadata.model_path) != current_path]
        retained.append(saved_artifact)
        return retained

    def _track_bundle(self, bundle: ExperimentResultBundle) -> None:
        if bundle.config.mlflow_tracking_mode == MLflowTrackingMode.MANUAL:
            tracker = self._build_tracker(MLflowExperimentTracker)
            if tracker is not None:
                run_id, tracking_warnings = tracker.log_bundle(
                    bundle,
                    existing_run_id=bundle.mlflow_run_id,
                )
                bundle.mlflow_run_id = run_id
                self._append_warnings(bundle, tracking_warnings)

        if self._metadata_store is not None:
            record_experiment_job(self._metadata_store, bundle)

    def _refresh_artifacts(self, bundle: ExperimentResultBundle) -> None:
        if self._artifacts_dir is None:
            return
        bundle.artifacts = write_experiment_artifacts(bundle, self._artifacts_dir)

    def _append_warnings(self, bundle: ExperimentResultBundle, warnings: list[str]) -> None:
        self._append_bundle_warnings(bundle, warnings)

    def _default_compare_metric(self, task_type: ExperimentTaskType) -> str:
        if task_type == ExperimentTaskType.CLASSIFICATION:
            return self._classification_compare_metric
        return self._regression_compare_metric

    def _default_tune_metric(self, task_type: ExperimentTaskType) -> str:
        if task_type == ExperimentTaskType.CLASSIFICATION:
            return self._classification_tune_metric
        return self._regression_tune_metric

    def _resolve_artifacts_dir(self) -> Path:
        if self._artifacts_dir is None:
            raise PyCaretExecutionError("Experiment artifacts directory is not configured.")
        return self._artifacts_dir

    def _resolve_models_dir(self) -> Path:
        if self._models_dir is None:
            raise PyCaretExecutionError("Model artifacts directory is not configured.")
        return self._models_dir

    def _resolve_snapshots_dir(self) -> Path:
        if self._snapshots_dir is None:
            raise PyCaretExecutionError("Experiment snapshot directory is not configured.")
        return self._snapshots_dir

    def _require_runtime(self, bundle: ExperimentResultBundle) -> ExperimentRuntimeState:
        if bundle.runtime is None:
            raise PyCaretExecutionError("Experiment runtime state is unavailable.")
        return bundle.runtime

    def _metric_value(self, metrics: dict[str, object], metric_name: str) -> float | None:
        value = metrics.get(metric_name)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None