"""CSV and delimiter-aware text loader."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from app.ingestion.base import BaseLoader
from app.ingestion.errors import ParseFailureError, RemoteAccessError, UnsupportedSourceError
from app.ingestion.schemas import DatasetInputSpec
from app.ingestion.types import DELIMITED_FILE_SUFFIXES, IngestionSourceType
from app.ingestion.url_loader import fetch_url_bytes
from app.ingestion.utils import sniff_delimiter


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
        if row_limit is not None:
            read_kwargs["nrows"] = row_limit

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

                dataframe = pd.read_csv(path, **read_kwargs)
                source_details = {
                    "source_kind": "path",
                    "delimiter": delimiter or "auto",
                    "encoding": input_spec.encoding,
                }
                return dataframe, source_details

            if input_spec.url is not None:
                payload, content_type, final_url = fetch_url_bytes(input_spec.url)
                sample_text = payload.decode(input_spec.encoding)
                if delimiter is None:
                    delimiter = sniff_delimiter(sample_text)
                    if delimiter is not None:
                        read_kwargs["sep"] = delimiter
                        read_kwargs.pop("engine", None)

                dataframe = pd.read_csv(StringIO(sample_text), **read_kwargs)
                source_details = {
                    "source_kind": "url",
                    "final_url": final_url,
                    "content_type": content_type,
                    "delimiter": delimiter or "auto",
                    "encoding": input_spec.encoding,
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


