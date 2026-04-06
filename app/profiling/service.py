"""High-level profiling service – the main entry point for profiling."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.config.models import ProfilingMode
from app.ingestion.schemas import LoadedDataset
from app.profiling.schemas import ProfilingArtifactBundle, ProfilingConfig, ProfilingResultSummary
from app.profiling.ydata_runner import YDataProfilingService
from app.storage import AppMetadataStore, record_profiling_job

logger = logging.getLogger(__name__)


def profile_dataset(
    df: pd.DataFrame,
    *,
    mode: ProfilingMode | None = None,
    dataset_name: str | None = None,
    artifacts_dir: Path | None = None,
    loaded_dataset: LoadedDataset | None = None,
    metadata_store: AppMetadataStore | None = None,
    large_dataset_row_threshold: int = 50_000,
    large_dataset_col_threshold: int = 100,
    sampling_row_threshold: int = 200_000,
    sample_size: int = 50_000,
) -> tuple[ProfilingResultSummary, ProfilingArtifactBundle | None]:
    """Convenience function: profile a DataFrame and optionally write artifacts."""
    config = ProfilingConfig(
        mode=mode or ProfilingMode.STANDARD,
        large_dataset_row_threshold=large_dataset_row_threshold,
        large_dataset_col_threshold=large_dataset_col_threshold,
        sampling_row_threshold=sampling_row_threshold,
        sample_size=sample_size,
    )
    service = YDataProfilingService(artifacts_dir=artifacts_dir)
    summary, bundle = service.profile(df, config, dataset_name=dataset_name)
    if metadata_store is not None:
        record_profiling_job(
            metadata_store,
            summary,
            artifacts=bundle,
            loaded_dataset=loaded_dataset,
        )
    return summary, bundle
