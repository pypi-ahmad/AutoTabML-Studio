"""In-memory pandas DataFrame loader."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.ingestion.base import BaseLoader
from app.ingestion.errors import UnsupportedSourceError
from app.ingestion.schemas import DatasetInputSpec
from app.ingestion.types import IngestionSourceType


class DataFrameLoader(BaseLoader):
    """Accept a pandas DataFrame without mutating caller-owned data."""

    supported_source_types = (IngestionSourceType.DATAFRAME,)

    def load_raw_dataframe(
        self,
        input_spec: DatasetInputSpec,
        *,
        row_limit: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if input_spec.dataframe is None:
            raise UnsupportedSourceError("DataFrame ingestion requires a pandas DataFrame input.")

        dataframe = input_spec.dataframe.copy(deep=True)
        if row_limit is not None:
            dataframe = dataframe.head(row_limit).copy(deep=True)

        source_details = {
            "source_kind": "dataframe",
            "copied_input": True,
            "original_row_count": len(input_spec.dataframe),
            "original_column_count": len(input_spec.dataframe.columns),
        }
        return dataframe, source_details
