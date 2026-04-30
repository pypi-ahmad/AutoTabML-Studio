"""Helpers that map app workflows into local metadata-store records."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from app.ingestion.schemas import LoadedDataset
from app.storage.models import AppJobStatus, AppJobType, JobRecord
from app.storage.store import AppMetadataStore

if TYPE_CHECKING:
    from app.modeling.benchmark.schemas import BenchmarkResultBundle
    from app.modeling.pycaret.schemas import ExperimentResultBundle
    from app.prediction.schemas import PredictionHistoryEntry
    from app.profiling.schemas import ProfilingArtifactBundle, ProfilingResultSummary
    from app.validation.schemas import ValidationArtifactBundle, ValidationResultSummary

logger = logging.getLogger(__name__)


def _enrich_metadata_with_description(
    metadata: dict[str, Any],
    job_type: AppJobType,
    *,
    dataset_name: str | None = None,
    mlflow_run_id: str | None = None,
) -> dict[str, Any]:
    """Add a template MLflow description to the metadata dict."""
    try:
        from app.tracking.description_generator import generate_template_description

        metadata["mlflow_description"] = generate_template_description(
            job_type,
            dataset_name=dataset_name,
            metadata=metadata,
            mlflow_run_id=mlflow_run_id,
        )
    except (ImportError, ValueError, RuntimeError):
        logger.debug(
            "Could not generate template description for %s job",
            getattr(job_type, "value", job_type),
            exc_info=True,
        )
    return metadata


def ensure_dataset_record(
    store: AppMetadataStore | None,
    loaded_dataset: LoadedDataset | None,
    *,
    dataset_name: str | None = None,
) -> str | None:
    if store is None or loaded_dataset is None:
        return None
    return store.upsert_dataset_from_loaded(loaded_dataset, dataset_name=dataset_name)


def record_validation_job(
    store: AppMetadataStore,
    summary: ValidationResultSummary,
    *,
    artifacts: ValidationArtifactBundle | None = None,
    loaded_dataset: LoadedDataset | None = None,
) -> str:
    dataset_key = ensure_dataset_record(store, loaded_dataset, dataset_name=summary.dataset_name)
    metadata = {
        "row_count": summary.row_count,
        "column_count": summary.column_count,
        "passed_count": summary.passed_count,
        "warning_count": summary.warning_count,
        "failed_count": summary.failed_count,
    }
    _enrich_metadata_with_description(metadata, AppJobType.VALIDATION, dataset_name=summary.dataset_name)
    return store.record_job(
        JobRecord(
            job_id=f"validation::{summary.dataset_name or 'dataset'}::{summary.run_timestamp.strftime('%Y%m%dT%H%M%S')}",
            job_type=AppJobType.VALIDATION,
            status=AppJobStatus.FAILED if summary.has_failures else AppJobStatus.SUCCESS,
            dataset_key=dataset_key,
            dataset_name=summary.dataset_name,
            title=f"Validation · {summary.dataset_name or 'dataset'}",
            primary_artifact_path=(artifacts.summary_json_path if artifacts is not None else None),
            summary_path=(artifacts.summary_json_path if artifacts is not None else None),
            metadata=metadata,
            created_at=summary.run_timestamp,
            updated_at=summary.run_timestamp,
        )
    )


def record_profiling_job(
    store: AppMetadataStore,
    summary: ProfilingResultSummary,
    *,
    artifacts: ProfilingArtifactBundle | None = None,
    loaded_dataset: LoadedDataset | None = None,
) -> str:
    dataset_key = ensure_dataset_record(store, loaded_dataset, dataset_name=summary.dataset_name)
    metadata = {
        "row_count": summary.row_count,
        "column_count": summary.column_count,
        "report_mode": summary.report_mode.value,
        "sampling_applied": summary.sampling_applied,
        "sample_size_used": summary.sample_size_used,
    }
    _enrich_metadata_with_description(metadata, AppJobType.PROFILING, dataset_name=summary.dataset_name)
    return store.record_job(
        JobRecord(
            job_id=f"profiling::{summary.dataset_name or 'dataset'}::{summary.run_timestamp.strftime('%Y%m%dT%H%M%S')}",
            job_type=AppJobType.PROFILING,
            status=AppJobStatus.SUCCESS,
            dataset_key=dataset_key,
            dataset_name=summary.dataset_name,
            title=f"Profiling · {summary.dataset_name or 'dataset'}",
            primary_artifact_path=(artifacts.html_report_path if artifacts is not None else None),
            summary_path=(artifacts.summary_json_path if artifacts is not None else None),
            metadata=metadata,
            created_at=summary.run_timestamp,
            updated_at=summary.run_timestamp,
        )
    )


def record_benchmark_job(
    store: AppMetadataStore,
    bundle: BenchmarkResultBundle,
    *,
    loaded_dataset: LoadedDataset | None = None,
) -> str:
    summary = bundle.summary
    dataset_key = ensure_dataset_record(store, loaded_dataset, dataset_name=bundle.dataset_name)
    metadata = {
        "task_type": bundle.task_type.value,
        "ranking_metric": summary.ranking_metric,
        "best_model_name": summary.best_model_name,
        "best_score": summary.best_score,
        "warnings": list(bundle.warnings),
    }
    _enrich_metadata_with_description(
        metadata, AppJobType.BENCHMARK,
        dataset_name=bundle.dataset_name, mlflow_run_id=bundle.mlflow_run_id,
    )
    return store.record_job(
        JobRecord(
            job_id=f"benchmark::{bundle.dataset_name or 'dataset'}::{summary.run_timestamp.strftime('%Y%m%dT%H%M%S')}",
            job_type=AppJobType.BENCHMARK,
            status=AppJobStatus.SUCCESS,
            dataset_key=dataset_key,
            dataset_name=bundle.dataset_name,
            title=f"Benchmark · {bundle.dataset_name or 'dataset'}",
            mlflow_run_id=bundle.mlflow_run_id,
            primary_artifact_path=(bundle.artifacts.leaderboard_csv_path if bundle.artifacts is not None else None),
            summary_path=(bundle.artifacts.summary_json_path if bundle.artifacts is not None else None),
            metadata=metadata,
            created_at=summary.run_timestamp,
            updated_at=summary.run_timestamp,
        )
    )


def record_experiment_job(
    store: AppMetadataStore,
    bundle: ExperimentResultBundle,
    *,
    loaded_dataset: LoadedDataset | None = None,
) -> str:
    summary = bundle.summary
    dataset_key = ensure_dataset_record(store, loaded_dataset, dataset_name=bundle.dataset_name)
    primary_saved_model_path = None
    if bundle.saved_model_metadata is not None:
        primary_saved_model_path = bundle.saved_model_metadata.model_path
    elif bundle.saved_model_artifacts:
        primary_saved_model_path = bundle.saved_model_artifacts[0].metadata.model_path

    job_id_parts = [
        bundle.mlflow_run_id or "local",
        bundle.dataset_name or "dataset",
        summary.run_timestamp.strftime("%Y%m%dT%H%M%S"),
    ]
    record_id = store.record_job(
        JobRecord(
            job_id="experiment::" + "::".join(job_id_parts),
            job_type=AppJobType.EXPERIMENT,
            status=AppJobStatus.SUCCESS,
            dataset_key=dataset_key,
            dataset_name=bundle.dataset_name,
            title=f"Experiment · {bundle.dataset_name or 'dataset'}",
            mlflow_run_id=bundle.mlflow_run_id,
            primary_artifact_path=primary_saved_model_path,
            summary_path=(bundle.artifacts.summary_json_path if bundle.artifacts is not None else None),
            metadata=_enrich_metadata_with_description(
                {
                    "task_type": bundle.task_type.value,
                    "best_baseline_model_name": summary.best_baseline_model_name,
                    "tuned_model_name": summary.tuned_model_name,
                    "selected_model_name": summary.selected_model_name,
                    "saved_model_name": summary.saved_model_name,
                    "warnings": list(bundle.warnings),
                },
                AppJobType.EXPERIMENT,
                dataset_name=bundle.dataset_name,
                mlflow_run_id=bundle.mlflow_run_id,
            ),
            created_at=summary.run_timestamp,
            updated_at=datetime.now(timezone.utc),
        )
    )

    if bundle.saved_model_artifacts:
        for saved_artifact in bundle.saved_model_artifacts:
            store.upsert_saved_model_metadata(
                saved_artifact.metadata,
                metadata_path=saved_artifact.metadata_path,
            )
    elif bundle.saved_model_metadata is not None:
        metadata_path = bundle.artifacts.saved_model_metadata_path if bundle.artifacts is not None else None
        store.upsert_saved_model_metadata(bundle.saved_model_metadata, metadata_path=metadata_path)

    return record_id


def record_prediction_history_entry(
    store: AppMetadataStore,
    entry: PredictionHistoryEntry,
) -> str:
    metadata = {
        "mode": entry.mode.value,
        "model_source": entry.model_source.value,
        "model_identifier": entry.model_identifier,
        "task_type": entry.task_type.value,
        "row_count": entry.row_count,
        "metadata_json_path": str(entry.metadata_json_path) if entry.metadata_json_path is not None else None,
    }
    _enrich_metadata_with_description(metadata, AppJobType.PREDICTION, dataset_name=entry.input_source)
    return store.record_job(
        JobRecord(
            job_id=entry.job_id,
            job_type=AppJobType.PREDICTION,
            status=AppJobStatus.SUCCESS if entry.status.value == "success" else AppJobStatus.FAILED,
            dataset_name=entry.input_source,
            title=f"Prediction · {entry.model_identifier}",
            primary_artifact_path=entry.output_artifact_path,
            summary_path=entry.summary_json_path,
            metadata=metadata,
            created_at=entry.timestamp,
            updated_at=entry.timestamp,
        )
    )