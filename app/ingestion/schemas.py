"""Pydantic schemas for tabular ingestion inputs and outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ingestion.types import IngestionSourceType


class DatasetInputSpec(BaseModel):
    """Canonical input contract for all supported ingestion paths."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_type: IngestionSourceType
    path: Path | None = None
    url: str | None = None
    dataframe: pd.DataFrame | None = None
    delimiter: str | None = None
    encoding: str = "utf-8"
    excel_sheet: str | int | None = None
    html_table_index: int = 0
    html_match_text: str | None = None
    kaggle_dataset_ref: str | None = None
    kaggle_file_name: str | None = None
    uci_id: int | None = None
    uci_name: str | None = None
    display_name: str | None = None
    project_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    row_limit: int | None = Field(default=None, gt=0)
    preview_rows: int | None = Field(default=None, gt=0)

    @property
    def locator(self) -> str:
        """Return the human-readable locator for lineage and logging."""

        if self.path is not None:
            return str(self.path)
        if self.url is not None:
            return self.url
        if self.kaggle_dataset_ref is not None:
            return self.kaggle_dataset_ref
        if self.uci_id is not None:
            return f"uci:{self.uci_id}"
        if self.uci_name is not None:
            return f"uci:{self.uci_name}"
        return "<in-memory-dataframe>"

    @model_validator(mode="after")
    def validate_required_fields(self) -> DatasetInputSpec:
        """Validate source-specific input requirements."""

        if self.source_type in {
            IngestionSourceType.CSV,
            IngestionSourceType.DELIMITED_TEXT,
            IngestionSourceType.EXCEL,
        } and not (self.path or self.url):
            raise ValueError(
                f"{self.source_type.value} ingestion requires a local path or URL."
            )

        if self.source_type == IngestionSourceType.HTML_TABLE and not self.url:
            raise ValueError("HTML table ingestion requires a URL.")

        if self.source_type == IngestionSourceType.URL_FILE and not self.url:
            raise ValueError("URL file ingestion requires a URL.")

        if self.source_type == IngestionSourceType.DATAFRAME and self.dataframe is None:
            raise ValueError("DataFrame ingestion requires a pandas DataFrame.")

        if self.source_type == IngestionSourceType.KAGGLE and not self.kaggle_dataset_ref:
            raise ValueError("Kaggle ingestion requires a dataset reference.")

        if self.source_type == IngestionSourceType.UCI_REPO:
            if self.uci_id is None and not self.uci_name:
                raise ValueError("UCI Repository ingestion requires a dataset ID or name.")
            if self.uci_id is not None and self.uci_name:
                raise ValueError("UCI Repository ingestion accepts either a dataset ID or a name, not both.")

        return self


class DatasetMetadata(BaseModel):
    """Metadata extracted during and after ingestion."""

    source_type: IngestionSourceType
    source_locator: str
    display_name: str | None = None
    project_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    file_extension: str | None = None
    sheet_name: str | int | None = None
    ingestion_timestamp: datetime
    row_count: int
    column_count: int
    column_names: list[str]
    dtype_summary: dict[str, str]
    memory_usage_bytes: int | None = None
    schema_hash: str
    content_hash: str | None = None
    normalization_actions: list[str] = Field(default_factory=list)
    source_details: dict[str, Any] = Field(default_factory=dict)
    is_preview: bool = False
    applied_row_limit: int | None = None


class LoadedDataset(BaseModel):
    """Loaded and normalized tabular dataset returned by ingestion."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: pd.DataFrame
    metadata: DatasetMetadata
    input_spec: DatasetInputSpec | None = None

    def preview(self, rows: int = 5) -> pd.DataFrame:
        """Return a safe preview copy of the first *rows* records."""

        return self.dataframe.head(rows).copy()
