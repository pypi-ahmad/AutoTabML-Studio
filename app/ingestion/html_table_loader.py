"""HTML table extraction loader for URL-based datasets."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from lxml import etree
import pandas as pd

from app.ingestion.base import BaseLoader
from app.ingestion.errors import ParseFailureError, UnsupportedSourceError
from app.ingestion.schemas import DatasetInputSpec
from app.ingestion.types import IngestionSourceType
from app.ingestion.url_loader import async_fetch_url_to_temp_file, fetch_url_to_temp_file


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
            with fetch_url_to_temp_file(
                input_spec.url,
                timeout=input_spec.url_timeout_seconds or 30.0,
                max_download_bytes=input_spec.url_max_download_bytes,
                max_redirects=input_spec.url_max_redirects,
                max_retries=input_spec.url_max_retries,
            ) as downloaded:
                dataframe, detected_table_count = self._read_html_table(
                    downloaded.path,
                    table_index=input_spec.html_table_index,
                    match_text=input_spec.html_match_text,
                    row_limit=row_limit,
                )
                if detected_table_count == 0:
                    raise ParseFailureError("No HTML tables were found at the supplied URL.")

                if dataframe is None:
                    raise ParseFailureError(
                        f"Requested HTML table index {input_spec.html_table_index} but only {detected_table_count} table(s) were found."
                    )

                source_details = {
                    "source_kind": "url",
                    "content_type": downloaded.content_type,
                    "final_url": downloaded.final_url,
                    "file_size_bytes": downloaded.file_size_bytes,
                    "detected_table_count": detected_table_count,
                    "selected_table_index": input_spec.html_table_index,
                    "match_text": input_spec.html_match_text,
                    "load_strategy": "streamed_temp_file_bounded_html_parse",
                }
                return dataframe, source_details
        except ImportError as exc:
            raise ParseFailureError(
                "HTML table parsing requires the 'lxml' parser dependency. Install the project dependencies and try again."
            ) from exc
        except etree.XMLSyntaxError as exc:
            raise ParseFailureError(f"Failed to parse HTML tables: {exc}") from exc
        except ValueError as exc:
            message = str(exc)
            if "No tables found" in message:
                raise ParseFailureError(
                    "No HTML tables were found at the supplied URL. Use a direct file URL or a page with <table> elements."
                ) from exc
            raise ParseFailureError(f"Failed to parse HTML tables: {exc}") from exc

    async def load_raw_dataframe_async(
        self,
        input_spec: DatasetInputSpec,
        *,
        row_limit: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not input_spec.url:
            raise UnsupportedSourceError("HTML table ingestion requires a URL.")

        try:
            async with async_fetch_url_to_temp_file(
                input_spec.url,
                timeout=input_spec.url_timeout_seconds or 30.0,
                max_download_bytes=input_spec.url_max_download_bytes,
                max_redirects=input_spec.url_max_redirects,
                max_retries=input_spec.url_max_retries,
            ) as downloaded:
                dataframe, detected_table_count = await asyncio.to_thread(
                    self._read_html_table,
                    downloaded.path,
                    table_index=input_spec.html_table_index,
                    match_text=input_spec.html_match_text,
                    row_limit=row_limit,
                )
                if detected_table_count == 0:
                    raise ParseFailureError("No HTML tables were found at the supplied URL.")

                if dataframe is None:
                    raise ParseFailureError(
                        f"Requested HTML table index {input_spec.html_table_index} but only {detected_table_count} table(s) were found."
                    )

                source_details = {
                    "source_kind": "url",
                    "content_type": downloaded.content_type,
                    "final_url": downloaded.final_url,
                    "file_size_bytes": downloaded.file_size_bytes,
                    "detected_table_count": detected_table_count,
                    "selected_table_index": input_spec.html_table_index,
                    "match_text": input_spec.html_match_text,
                    "load_strategy": "streamed_temp_file_bounded_html_parse_async",
                }
                return dataframe, source_details
        except ImportError as exc:
            raise ParseFailureError(
                "HTML table parsing requires the 'lxml' parser dependency. Install the project dependencies and try again."
            ) from exc
        except etree.XMLSyntaxError as exc:
            raise ParseFailureError(f"Failed to parse HTML tables: {exc}") from exc
        except ValueError as exc:
            message = str(exc)
            if "No tables found" in message:
                raise ParseFailureError(
                    "No HTML tables were found at the supplied URL. Use a direct file URL or a page with <table> elements."
                ) from exc
            raise ParseFailureError(f"Failed to parse HTML tables: {exc}") from exc

    def _read_html_table(
        self,
        path,
        *,
        table_index: int,
        match_text: str | None,
        row_limit: int | None,
    ) -> tuple[pd.DataFrame | None, int]:
        compiled_match = re.compile(match_text, re.IGNORECASE) if match_text else None
        parser = etree.HTMLPullParser(events=("start", "end"), recover=True)

        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                parser.feed(chunk)

        matched_table_count = 0
        selected_frame: pd.DataFrame | None = None
        inside_target_table = False
        nested_table_depth = 0
        current_header: list[str] | None = None
        current_rows: list[list[Any]] = []
        current_row: list[str] | None = None
        current_row_tags: list[str] | None = None
        table_matches = compiled_match is None
        current_table_is_selected = False

        for event, element in parser.read_events():
            tag = self._normalize_tag(element.tag)
            if tag is None:
                continue

            if event == "start" and tag == "table":
                if inside_target_table:
                    nested_table_depth += 1
                    continue

                inside_target_table = True
                nested_table_depth = 0
                current_header = None
                current_rows = []
                current_row = None
                current_row_tags = None
                table_matches = compiled_match is None
                current_table_is_selected = matched_table_count == table_index
                continue

            if not inside_target_table:
                continue

            if event == "start" and tag == "tr" and nested_table_depth == 0:
                current_row = []
                current_row_tags = []
                continue

            if event == "end" and tag == "table":
                if nested_table_depth > 0:
                    nested_table_depth -= 1
                    self._clear_element(element)
                    continue

                if table_matches:
                    if current_table_is_selected:
                        selected_frame = self._build_dataframe(current_header, current_rows)
                    matched_table_count += 1

                inside_target_table = False
                current_header = None
                current_rows = []
                current_row = None
                current_row_tags = None
                table_matches = compiled_match is None
                current_table_is_selected = False
                self._clear_element(element)
                continue

            if nested_table_depth > 0:
                continue

            if event == "end" and tag in {"th", "td"} and current_row is not None and current_row_tags is not None:
                cell_text = self._extract_text(element)
                current_row.append(cell_text)
                current_row_tags.append(tag)
                if compiled_match is not None and not table_matches and compiled_match.search(cell_text):
                    table_matches = True
                self._clear_element(element)
                continue

            if event == "end" and tag == "caption":
                caption_text = self._extract_text(element)
                if compiled_match is not None and not table_matches and compiled_match.search(caption_text):
                    table_matches = True
                self._clear_element(element)
                continue

            if event == "end" and tag == "tr" and current_row is not None and current_row_tags is not None:
                row_text = " ".join(value for value in current_row if value)
                if compiled_match is not None and not table_matches and compiled_match.search(row_text):
                    table_matches = True

                if self._is_header_row(current_row_tags, current_header):
                    current_header = current_row
                elif current_table_is_selected and (row_limit is None or len(current_rows) < row_limit):
                    current_rows.append(current_row)

                current_row = None
                current_row_tags = None
                self._clear_element(element)

        return selected_frame, matched_table_count

    def _build_dataframe(self, header: list[str] | None, rows: list[list[Any]]) -> pd.DataFrame:
        if header is None:
            if not rows:
                return pd.DataFrame()
            column_count = max(len(row) for row in rows)
            normalized_rows = [row + [None] * (column_count - len(row)) for row in rows]
            return self._coerce_dataframe_types(pd.DataFrame(normalized_rows))

        column_count = max(len(header), max((len(row) for row in rows), default=0))
        normalized_header = header + [f"Unnamed: {index}" for index in range(len(header), column_count)]
        normalized_rows = [row + [None] * (column_count - len(row)) for row in rows]
        return self._coerce_dataframe_types(pd.DataFrame(normalized_rows, columns=normalized_header))

    def _is_header_row(self, row_tags: list[str], existing_header: list[str] | None) -> bool:
        return existing_header is None and row_tags and all(tag == "th" for tag in row_tags)

    def _extract_text(self, element) -> str:
        return " ".join(part.strip() for part in element.itertext() if part and part.strip())

    def _normalize_tag(self, tag: Any) -> str | None:
        if not isinstance(tag, str):
            return None
        if "}" in tag:
            tag = tag.rsplit("}", maxsplit=1)[-1]
        return tag.lower()

    def _coerce_dataframe_types(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for column_name in dataframe.columns:
            try:
                dataframe[column_name] = pd.to_numeric(dataframe[column_name])
            except (TypeError, ValueError):
                continue
        return dataframe

    def _clear_element(self, element) -> None:
        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]
