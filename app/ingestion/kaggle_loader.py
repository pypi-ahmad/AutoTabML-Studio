"""Kaggle dataset loader with explicit, realistic boundaries."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from app.ingestion.base import BaseLoader
from app.ingestion.errors import IngestionError, RemoteAccessError, UnsupportedSourceError
from app.ingestion.schemas import DatasetInputSpec
from app.ingestion.types import DELIMITED_FILE_SUFFIXES, EXCEL_FILE_SUFFIXES, IngestionSourceType


class KaggleLoader(BaseLoader):
    """Download a Kaggle dataset locally, then delegate to a concrete file loader."""

    supported_source_types = (IngestionSourceType.KAGGLE,)

    def load_raw_dataframe(
        self,
        input_spec: DatasetInputSpec,
        *,
        row_limit: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        dataset_ref = input_spec.kaggle_dataset_ref
        if not dataset_ref:
            raise UnsupportedSourceError("Kaggle ingestion requires a dataset reference like 'owner/dataset-name'.")

        api = self._build_kaggle_api()

        try:
            with tempfile.TemporaryDirectory(prefix="autotabml-kaggle-") as temp_dir:
                api.dataset_download_files(dataset_ref, path=temp_dir, unzip=True, quiet=True)
                selected_path = self._select_dataset_file(Path(temp_dir), input_spec.kaggle_file_name)
                delegated_spec = input_spec.model_copy(
                    update={
                        "path": selected_path,
                        "url": None,
                        "source_type": self._infer_local_source_type(selected_path),
                    }
                )

                if delegated_spec.source_type == IngestionSourceType.EXCEL:
                    from app.ingestion.excel_loader import ExcelLoader

                    dataframe, source_details = ExcelLoader().load_raw_dataframe(delegated_spec, row_limit=row_limit)
                else:
                    from app.ingestion.csv_loader import CSVLoader

                    dataframe, source_details = CSVLoader().load_raw_dataframe(delegated_spec, row_limit=row_limit)

                source_details.update(
                    {
                        "source_kind": "kaggle",
                        "kaggle_dataset_ref": dataset_ref,
                        "kaggle_selected_file": selected_path.name,
                        "kaggle_note": "Kaggle ingestion downloads the dataset archive locally before loading one tabular file.",
                    }
                )
                return dataframe, source_details
        except (OSError, RuntimeError) as exc:
            raise RemoteAccessError(f"Failed to download or unpack Kaggle dataset '{dataset_ref}': {exc}") from exc

    def _build_kaggle_api(self):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:
            raise RemoteAccessError(
                "Kaggle ingestion requires the optional 'kaggle' package. Install it with `pip install .[kaggle]`."
            ) from exc

        if not self._has_kaggle_credentials():
            raise RemoteAccessError(
                "Kaggle credentials are not configured. Set KAGGLE_USERNAME and KAGGLE_KEY, "
                "or place kaggle.json in ~/.kaggle before using Kaggle ingestion."
            )

        api = KaggleApi()
        api.authenticate()
        return api

    def _has_kaggle_credentials(self) -> bool:
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return True

        config_path = Path.home() / ".kaggle" / "kaggle.json"
        return config_path.exists()

    def _select_dataset_file(self, directory: Path, explicit_file_name: str | None) -> Path:
        supported_files = [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in DELIMITED_FILE_SUFFIXES.union(EXCEL_FILE_SUFFIXES)
        ]

        if explicit_file_name:
            for candidate in supported_files:
                if candidate.name == explicit_file_name:
                    return candidate
            raise UnsupportedSourceError(
                f"Kaggle dataset file '{explicit_file_name}' was not found. Available supported files: "
                + ", ".join(sorted(path.name for path in supported_files))
            )

        if not supported_files:
            raise UnsupportedSourceError(
                "The Kaggle dataset archive did not contain a supported tabular file. "
                "Supported types are CSV, TSV, TXT, XLSX, XLS, XLSM, and XLSB."
            )

        if len(supported_files) > 1:
            raise IngestionError(
                "The Kaggle dataset archive contains multiple supported tabular files. "
                "Set kaggle_file_name explicitly to choose one file."
            )

        return supported_files[0]

    def _infer_local_source_type(self, path: Path) -> IngestionSourceType:
        suffix = path.suffix.lower()
        if suffix in EXCEL_FILE_SUFFIXES:
            return IngestionSourceType.EXCEL
        return IngestionSourceType.CSV if suffix == ".csv" else IngestionSourceType.DELIMITED_TEXT