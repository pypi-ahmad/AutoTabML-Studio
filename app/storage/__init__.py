"""Local metadata storage exports."""

from app.config.models import AppSettings
from app.storage.models import (
    AppJobStatus,
    AppJobType,
    BatchItemStatus,
    BatchRunItemRecord,
    BatchRunRecord,
    BatchRunStatus,
    DatasetRecord,
    JobRecord,
    ProjectRecord,
    SavedLocalModelRecord,
)
from app.storage.recorders import (
    ensure_dataset_record,
    record_benchmark_job,
    record_experiment_job,
    record_prediction_history_entry,
    record_profiling_job,
    record_validation_job,
)
from app.storage.store import AppMetadataStore


def build_metadata_store(settings: AppSettings) -> AppMetadataStore | None:
    """Return the local app metadata store for the supplied settings when configured."""

    database_settings = getattr(settings, "database", None)
    database_path = getattr(database_settings, "path", None)
    if database_path is None:
        return None

    store = AppMetadataStore(database_path)
    store.initialize_if_needed()
    return store

__all__ = [
    "AppJobStatus",
    "AppJobType",
    "AppMetadataStore",
    "BatchItemStatus",
    "BatchRunItemRecord",
    "BatchRunRecord",
    "BatchRunStatus",
    "DatasetRecord",
    "JobRecord",
    "ProjectRecord",
    "SavedLocalModelRecord",
    "build_metadata_store",
    "ensure_dataset_record",
    "record_benchmark_job",
    "record_experiment_job",
    "record_prediction_history_entry",
    "record_profiling_job",
    "record_validation_job",
]