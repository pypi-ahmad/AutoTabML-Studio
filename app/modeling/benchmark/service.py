"""Convenience entry point for benchmark execution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config.enums import ExecutionBackend, WorkspaceMode
from app.ingestion.schemas import LoadedDataset
from app.modeling.benchmark.lazypredict_runner import LazyPredictBenchmarkService
from app.modeling.benchmark.schemas import BenchmarkConfig, BenchmarkResultBundle
from app.storage import AppMetadataStore, record_benchmark_job


def benchmark_dataset(
    df: pd.DataFrame,
    config: BenchmarkConfig,
    *,
    dataset_name: str | None = None,
    dataset_fingerprint: str | None = None,
    loaded_dataset: LoadedDataset | None = None,
    metadata_store: AppMetadataStore | None = None,
    execution_backend: ExecutionBackend = ExecutionBackend.LOCAL,
    workspace_mode: WorkspaceMode | None = None,
    artifacts_dir: Path | None = None,
    classification_default_metric: str = "Balanced Accuracy",
    regression_default_metric: str = "Adjusted R-Squared",
    sampling_row_threshold: int = 100_000,
    suggested_sample_rows: int = 50_000,
    mlflow_experiment_name: str | None = None,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> BenchmarkResultBundle:
    """Run a benchmark with the LazyPredict benchmark service."""

    service = LazyPredictBenchmarkService(
        artifacts_dir=artifacts_dir,
        classification_default_metric=classification_default_metric,
        regression_default_metric=regression_default_metric,
        sampling_row_threshold=sampling_row_threshold,
        suggested_sample_rows=suggested_sample_rows,
        mlflow_experiment_name=mlflow_experiment_name,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    bundle = service.run(
        df,
        config,
        dataset_name=dataset_name,
        dataset_fingerprint=dataset_fingerprint,
        execution_backend=execution_backend,
        workspace_mode=workspace_mode,
    )
    if metadata_store is not None:
        record_benchmark_job(
            metadata_store,
            bundle,
            loaded_dataset=loaded_dataset,
        )
    return bundle