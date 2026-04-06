"""HTML table extraction loader for URL-based datasets."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd

from app.ingestion.base import BaseLoader
from app.ingestion.errors import ParseFailureError, UnsupportedSourceError
from app.ingestion.schemas import DatasetInputSpec
from app.ingestion.types import IngestionSourceType
from app.ingestion.url_loader import fetch_url_text


class HTMLTableLoader(BaseLoader):
    """Extract HTML tables from a URL and return one table as a DataFrame."""

    supported_source_types = (IngestionSourceType.HTML_TABLE,)

    def load_raw_dataframe(
        self,
        input_spec: DatasetInputSpec,
        *,
        row_limit: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not input_spec.url:
            raise UnsupportedSourceError("HTML table ingestion requires a URL.")

        try:
            html_text, content_type, final_url = fetch_url_text(input_spec.url, encoding=input_spec.encoding)
            read_kwargs: dict[str, Any] = {"flavor": ["lxml"]}
            if input_spec.html_match_text:
                read_kwargs["match"] = input_spec.html_match_text

            tables = pd.read_html(StringIO(html_text), **read_kwargs)
            if not tables:
                raise ParseFailureError("No HTML tables were found at the supplied URL.")

            if input_spec.html_table_index >= len(tables):
                raise ParseFailureError(
                    f"Requested HTML table index {input_spec.html_table_index} but only {len(tables)} table(s) were found."
                )

            dataframe = tables[input_spec.html_table_index]
            if row_limit is not None:
                dataframe = dataframe.head(row_limit)

            source_details = {
                "source_kind": "url",
                "content_type": content_type,
                "final_url": final_url,
                "detected_table_count": len(tables),
                "selected_table_index": input_spec.html_table_index,
                "match_text": input_spec.html_match_text,
            }
            return dataframe, source_details
        except ImportError as exc:
            raise ParseFailureError(
                "HTML table parsing requires the 'lxml' parser dependency. Install the project dependencies and try again."
            ) from exc
        except ValueError as exc:
            message = str(exc)
            if "No tables found" in message:
                raise ParseFailureError(
                    "No HTML tables were found at the supplied URL. Use a direct file URL or a page with <table> elements."
                ) from exc
            raise ParseFailureError(f"Failed to parse HTML tables: {exc}") from exc
