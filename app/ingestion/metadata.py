"""Metadata extraction and deterministic hashing for ingested datasets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from app.ingestion.schemas import DatasetInputSpec, DatasetMetadata


def compute_schema_hash(dataframe: pd.DataFrame) -> str:
    """Return a deterministic hash of column names and dtypes."""

    payload = [
        {"name": str(column), "dtype": str(dtype)}
        for column, dtype in zip(dataframe.columns, dataframe.dtypes, strict=True)
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_content_hash(dataframe: pd.DataFrame, sample_rows: int = 50) -> str | None:
    """Return a lightweight deterministic hash of the first *sample_rows* rows."""

    if dataframe.empty:
        return None

    sample = dataframe.head(sample_rows).reset_index(drop=True)
    payload = {
        "columns": [str(column) for column in sample.columns],
        "rows": [
            [_stable_value(value) for value in row]
            for row in sample.itertuples(index=False, name=None)
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_dataset_metadata(
    dataframe: pd.DataFrame,
    input_spec: DatasetInputSpec,
    *,
    normalization_actions: list[str] | None = None,
    source_details: dict[str, Any] | None = None,
    is_preview: bool = False,
    applied_row_limit: int | None = None,
) -> DatasetMetadata:
    """Build rich dataset metadata for lineage and future downstream workflows."""

    normalization_actions = normalization_actions or []
    source_details = source_details or {}

    return DatasetMetadata(
        source_type=input_spec.source_type,
        source_locator=input_spec.locator,
        display_name=input_spec.display_name,
        project_id=input_spec.project_id,
        tags=list(input_spec.tags),
        file_extension=detect_file_extension(input_spec),
        sheet_name=source_details.get("sheet_name"),
        ingestion_timestamp=datetime.now(timezone.utc),
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        column_names=[str(column) for column in dataframe.columns.tolist()],
        dtype_summary={str(column): str(dtype) for column, dtype in dataframe.dtypes.items()},
        memory_usage_bytes=int(dataframe.memory_usage(deep=True).sum()),
        schema_hash=compute_schema_hash(dataframe),
        content_hash=compute_content_hash(dataframe),
        normalization_actions=normalization_actions,
        source_details=source_details,
        is_preview=is_preview,
        applied_row_limit=applied_row_limit,
    )


def detect_file_extension(input_spec: DatasetInputSpec) -> str | None:
    """Infer the file extension from a local path or URL, if present."""

    if input_spec.path is not None:
        suffix = Path(input_spec.path).suffix.lower()
        return suffix or None

    if input_spec.url is not None:
        parsed = urlparse(input_spec.url)
        suffix = Path(parsed.path).suffix.lower()
        return suffix or None

    if input_spec.kaggle_file_name:
        suffix = Path(input_spec.kaggle_file_name).suffix.lower()
        return suffix or None

    return None


def _stable_value(value: Any) -> Any:
    """Convert values to a stable, JSON-serializable representation."""

    if pd.isna(value):
        return "<NA>"

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)
