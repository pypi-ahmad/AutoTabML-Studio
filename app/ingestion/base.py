"""Base abstractions shared by all ingestion loaders."""

from __future__ import annotations

import asyncio
import abc
from typing import Any

import pandas as pd

from app.ingestion.errors import UnsupportedSourceError
from app.ingestion.metadata import extract_dataset_metadata
from app.ingestion.normalizer import normalize_to_pandas
from app.ingestion.schemas import DatasetInputSpec, LoadedDataset
from app.ingestion.types import IngestionSourceType


class BaseLoader(abc.ABC):
    """Common load lifecycle for all dataset sources."""

    supported_source_types: tuple[IngestionSourceType, ...] = tuple()

    def load(self, input_spec: DatasetInputSpec, *, preview: bool = False, preview_rows: int | None = None) -> LoadedDataset:
        """Load, normalize, and enrich a dataset from the supplied input spec."""

        self.validate_input(input_spec)
        applied_row_limit = preview_rows if preview else input_spec.row_limit
        raw_dataframe, source_details = self.load_raw_dataframe(
            input_spec,
            row_limit=applied_row_limit,
        )
        normalized_dataframe, actions = normalize_to_pandas(raw_dataframe)
        metadata = extract_dataset_metadata(
            normalized_dataframe,
            input_spec,
            normalization_actions=actions,
            source_details=source_details,
            is_preview=preview,
            applied_row_limit=applied_row_limit,
        )
        return LoadedDataset(
            dataframe=normalized_dataframe,
            metadata=metadata,
            input_spec=input_spec,
        )

    async def load_async(
        self,
        input_spec: DatasetInputSpec,
        *,
        preview: bool = False,
        preview_rows: int | None = None,
    ) -> LoadedDataset:
        """Async counterpart of :meth:`load` for I/O-heavy loader implementations."""

        self.validate_input(input_spec)
        applied_row_limit = preview_rows if preview else input_spec.row_limit
        raw_dataframe, source_details = await self.load_raw_dataframe_async(
            input_spec,
            row_limit=applied_row_limit,
        )
        normalized_dataframe, actions = await asyncio.to_thread(
            normalize_to_pandas,
            raw_dataframe,
        )
        metadata = await asyncio.to_thread(
            extract_dataset_metadata,
            normalized_dataframe,
            input_spec,
            normalization_actions=actions,
            source_details=source_details,
            is_preview=preview,
            applied_row_limit=applied_row_limit,
        )
        return LoadedDataset(
            dataframe=normalized_dataframe,
            metadata=metadata,
            input_spec=input_spec,
        )

    def preview(self, input_spec: DatasetInputSpec, rows: int = 5) -> LoadedDataset:
        """Load only a preview slice where the underlying loader supports it."""

        return self.load(input_spec, preview=True, preview_rows=rows)

    async def preview_async(self, input_spec: DatasetInputSpec, rows: int = 5) -> LoadedDataset:
        """Async counterpart of :meth:`preview`."""

        return await self.load_async(input_spec, preview=True, preview_rows=rows)

    def validate_input(self, input_spec: DatasetInputSpec) -> None:
        """Ensure the loader is being used for a compatible source type."""

        if self.supported_source_types and input_spec.source_type not in self.supported_source_types:
            supported = ", ".join(source_type.value for source_type in self.supported_source_types)
            raise UnsupportedSourceError(
                f"{self.__class__.__name__} does not support source type '{input_spec.source_type.value}'. "
                f"Supported types: {supported}."
            )

    @abc.abstractmethod
    def load_raw_dataframe(
        self,
        input_spec: DatasetInputSpec,
        *,
        row_limit: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return a raw DataFrame and source details before normalization."""

    async def load_raw_dataframe_async(
        self,
        input_spec: DatasetInputSpec,
        *,
        row_limit: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Async counterpart of :meth:`load_raw_dataframe`.

        Loaders with native async I/O should override this. The default falls
        back to a worker thread so existing sync loaders remain usable via the
        async ingestion API.
        """

        return await asyncio.to_thread(
            self.load_raw_dataframe,
            input_spec,
            row_limit=row_limit,
        )
