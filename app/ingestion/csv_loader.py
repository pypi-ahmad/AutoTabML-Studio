"""CSV and delimiter-aware text loader."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from app.ingestion.base import BaseLoader
from app.ingestion.errors import ParseFailureError, RemoteAccessError, UnsupportedSourceError
from app.ingestion.schemas import DatasetInputSpec
from app.ingestion.types import DELIMITED_FILE_SUFFIXES, IngestionSourceType
from app.ingestion.url_loader import (
    async_fetch_url_to_temp_file,
    ensure_local_file_size_guard,
    fetch_url_to_temp_file,
)
from app.ingestion.utils import sniff_delimiter

_CSV_CHUNK_SIZE_ROWS = 50_000
_CSV_SNIFF_BYTES = 8192


class CSVLoader(BaseLoader):
    """Load local or remote delimited tabular files into pandas."""

    supported_source_types = (
        IngestionSourceType.CSV,
        IngestionSourceType.DELIMITED_TEXT,
    )

    def load_raw_dataframe(
        self,
        input_spec: DatasetInputSpec,
        *,
        row_limit: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        delimiter = self._resolve_delimiter(input_spec)
        read_kwargs: dict[str, Any] = {
            "encoding": input_spec.encoding,
            "on_bad_lines": "error",
        }

        if delimiter is None:
            read_kwargs["sep"] = None
            read_kwargs["engine"] = "python"
        else:
            read_kwargs["sep"] = delimiter

        try:
            if input_spec.path is not None:
                path = Path(input_spec.path)
                if not path.exists():
                    raise UnsupportedSourceError(f"Local file not found: {path}")

                file_size = ensure_local_file_size_guard(
                    path,
                    max_bytes=input_spec.local_max_file_bytes,
                )
                delimiter = self._update_delimiter_from_sample(path, input_spec, delimiter, read_kwargs)
                dataframe = self._read_csv_in_chunks(path, read_kwargs, row_limit=row_limit)
                source_details = {
                    "source_kind": "path",
                    "delimiter": delimiter or "auto",
                    "encoding": input_spec.encoding,
                    "file_size_bytes": file_size,
                    "load_strategy": "chunked_read_csv",
                }
                return dataframe, source_details

            if input_spec.url is not None:
                with fetch_url_to_temp_file(
                    input_spec.url,
                    timeout=input_spec.url_timeout_seconds or 30.0,
                    max_download_bytes=input_spec.url_max_download_bytes,
                    max_redirects=input_spec.url_max_redirects,
                    max_retries=input_spec.url_max_retries,
                ) as downloaded:
                    delimiter = self._update_delimiter_from_sample(
                        downloaded.path,
                        input_spec,
                        delimiter,
                        read_kwargs,
                    )
                    dataframe = self._read_csv_in_chunks(
                        downloaded.path,
                        read_kwargs,
                        row_limit=row_limit,
                    )
                    source_details = {
                        "source_kind": "url",
                        "final_url": downloaded.final_url,
                        "content_type": downloaded.content_type,
                        "delimiter": delimiter or "auto",
                        "encoding": input_spec.encoding,
                        "file_size_bytes": downloaded.file_size_bytes,
                        "load_strategy": "streamed_temp_file_chunked_read_csv",
                    }
                    return dataframe, source_details

            raise UnsupportedSourceError("CSV ingestion requires either a local path or a URL.")
        except UnicodeDecodeError as exc:
            raise ParseFailureError(
                "The delimited file could not be decoded with the configured encoding. "
                f"Tried encoding '{input_spec.encoding}'."
            ) from exc
        except pd.errors.ParserError as exc:
            raise ParseFailureError(
                "Failed to parse the delimited file. Check the delimiter, quoting, and file structure."
            ) from exc
        except OSError as exc:
            raise RemoteAccessError(f"Failed to access the dataset source: {exc}") from exc
        except UnsupportedSourceError:
            raise
        except ValueError as exc:
            raise ParseFailureError(f"Failed to load the delimited file: {exc}") from exc

    async def load_raw_dataframe_async(
        self,
        input_spec: DatasetInputSpec,
        *,
        row_limit: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if input_spec.path is not None:
            return await asyncio.to_thread(
                self.load_raw_dataframe,
                input_spec,
                row_limit=row_limit,
            )

        delimiter = self._resolve_delimiter(input_spec)
        read_kwargs: dict[str, Any] = {
            "encoding": input_spec.encoding,
            "on_bad_lines": "error",
        }

        if delimiter is None:
            read_kwargs["sep"] = None
            read_kwargs["engine"] = "python"
        else:
            read_kwargs["sep"] = delimiter

        try:
            if input_spec.url is not None:
                async with async_fetch_url_to_temp_file(
                    input_spec.url,
                    timeout=input_spec.url_timeout_seconds or 30.0,
                    max_download_bytes=input_spec.url_max_download_bytes,
                    max_redirects=input_spec.url_max_redirects,
                    max_retries=input_spec.url_max_retries,
                ) as downloaded:
                    delimiter = await asyncio.to_thread(
                        self._update_delimiter_from_sample,
                        downloaded.path,
                        input_spec,
                        delimiter,
                        read_kwargs,
                    )
                    dataframe = await asyncio.to_thread(
                        self._read_csv_in_chunks,
                        downloaded.path,
                        read_kwargs,
                        row_limit=row_limit,
                    )
                    source_details = {
                        "source_kind": "url",
                        "final_url": downloaded.final_url,
                        "content_type": downloaded.content_type,
                        "delimiter": delimiter or "auto",
                        "encoding": input_spec.encoding,
                        "file_size_bytes": downloaded.file_size_bytes,
                        "load_strategy": "streamed_temp_file_chunked_read_csv_async",
                    }
                    return dataframe, source_details

            raise UnsupportedSourceError("CSV ingestion requires either a local path or a URL.")
        except UnicodeDecodeError as exc:
            raise ParseFailureError(
                "The delimited file could not be decoded with the configured encoding. "
                f"Tried encoding '{input_spec.encoding}'."
            ) from exc
        except pd.errors.ParserError as exc:
            raise ParseFailureError(
                "Failed to parse the delimited file. Check the delimiter, quoting, and file structure."
            ) from exc
        except OSError as exc:
            raise RemoteAccessError(f"Failed to access the dataset source: {exc}") from exc
        except UnsupportedSourceError:
            raise
        except ValueError as exc:
            raise ParseFailureError(f"Failed to load the delimited file: {exc}") from exc

    def _resolve_delimiter(self, input_spec: DatasetInputSpec) -> str | None:
        if input_spec.delimiter:
            return input_spec.delimiter

        locator = input_spec.locator.lower()
        if locator.endswith(".tsv"):
            return "\t"
        if locator.endswith(".csv"):
            return ","
        if any(locator.endswith(suffix) for suffix in DELIMITED_FILE_SUFFIXES):
            return None
        return None

    def _update_delimiter_from_sample(
        self,
        path: Path,
        input_spec: DatasetInputSpec,
        delimiter: str | None,
        read_kwargs: dict[str, Any],
    ) -> str | None:
        if delimiter is not None:
            return delimiter

        with path.open("rb") as handle:
            sample_bytes = handle.read(_CSV_SNIFF_BYTES)
        sample_text = sample_bytes.decode(input_spec.encoding)
        detected = sniff_delimiter(sample_text)
        if detected is None:
            return delimiter

        read_kwargs["sep"] = detected
        read_kwargs.pop("engine", None)
        return detected

    def _read_csv_in_chunks(
        self,
        path: Path,
        read_kwargs: dict[str, Any],
        *,
        row_limit: int | None,
    ) -> pd.DataFrame:
        chunksize = min(row_limit, _CSV_CHUNK_SIZE_ROWS) if row_limit is not None else _CSV_CHUNK_SIZE_ROWS
        reader = pd.read_csv(path, chunksize=chunksize, **read_kwargs)

        collected_chunks: list[pd.DataFrame] = []
        remaining_rows = row_limit
        for chunk in reader:
            if remaining_rows is not None and len(chunk) > remaining_rows:
                chunk = chunk.head(remaining_rows)
            collected_chunks.append(chunk)

            if remaining_rows is None:
                continue

            remaining_rows -= len(chunk)
            if remaining_rows <= 0:
                break

        if not collected_chunks:
            return pd.read_csv(path, nrows=0, **read_kwargs)
        if len(collected_chunks) == 1:
            return collected_chunks[0].reset_index(drop=True)
        return pd.concat(collected_chunks, ignore_index=True)


