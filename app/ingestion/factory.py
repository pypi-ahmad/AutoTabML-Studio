"""Loader selection and public ingestion entry points."""

from __future__ import annotations

from app.ingestion.base import BaseLoader
from app.ingestion.schemas import DatasetInputSpec, LoadedDataset
from app.ingestion.types import IngestionSourceType


def get_loader(input_spec: DatasetInputSpec) -> BaseLoader:
    """Return the concrete loader for the supplied input spec."""

    if input_spec.source_type in {
        IngestionSourceType.CSV,
        IngestionSourceType.DELIMITED_TEXT,
    }:
        from app.ingestion.csv_loader import CSVLoader

        return CSVLoader()

    if input_spec.source_type == IngestionSourceType.EXCEL:
        from app.ingestion.excel_loader import ExcelLoader

        return ExcelLoader()

    if input_spec.source_type == IngestionSourceType.HTML_TABLE:
        from app.ingestion.html_table_loader import HTMLTableLoader

        return HTMLTableLoader()

    if input_spec.source_type == IngestionSourceType.DATAFRAME:
        from app.ingestion.dataframe_loader import DataFrameLoader

        return DataFrameLoader()

    if input_spec.source_type == IngestionSourceType.KAGGLE:
        from app.ingestion.kaggle_loader import KaggleLoader

        return KaggleLoader()

    if input_spec.source_type == IngestionSourceType.URL_FILE:
        from app.ingestion.url_loader import URLLoader

        return URLLoader()

    if input_spec.source_type == IngestionSourceType.UCI_REPO:
        from app.ingestion.uci_loader import UCIRepoLoader

        return UCIRepoLoader()

    raise ValueError(f"Unsupported source type: {input_spec.source_type.value}")


def load_dataset(input_spec: DatasetInputSpec) -> LoadedDataset:
    """Public entry point for full dataset loading."""

    return get_loader(input_spec).load(input_spec)


def preview_dataset(input_spec: DatasetInputSpec, rows: int = 5) -> LoadedDataset:
    """Public entry point for lightweight preview loading."""

    return get_loader(input_spec).preview(input_spec, rows=rows)