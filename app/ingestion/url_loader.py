"""URL routing and remote fetch helpers for ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pandas as pd

from app.ingestion.base import BaseLoader
from app.ingestion.errors import RemoteAccessError, UnsupportedSourceError
from app.ingestion.schemas import DatasetInputSpec
from app.ingestion.types import (
    DELIMITED_FILE_SUFFIXES,
    EXCEL_FILE_SUFFIXES,
    HTML_FILE_SUFFIXES,
    IngestionSourceType,
)

_HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}
_CSV_CONTENT_TYPES = {"text/csv", "application/csv"}
_TEXT_CONTENT_TYPES = {"text/plain"}
_EXCEL_CONTENT_TYPES = {
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel.sheet.binary.macroenabled.12",
}


@dataclass
class URLProbeResult:
    """Inference result for a remote URL source."""

    final_url: str
    content_type: str | None
    routed_source_type: IngestionSourceType
    delimiter: str | None = None
    probe_method: str = "head"


class URLLoader(BaseLoader):
    """Route a URL to the correct concrete loader."""

    supported_source_types = (IngestionSourceType.URL_FILE,)

    def load_raw_dataframe(
        self,
        input_spec: DatasetInputSpec,
        *,
        row_limit: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not input_spec.url:
            raise UnsupportedSourceError("URL file ingestion requires a URL.")

        probe = probe_url(input_spec.url)
        routed_spec = input_spec.model_copy(
            update={
                "source_type": probe.routed_source_type,
                "delimiter": input_spec.delimiter or probe.delimiter,
            }
        )

        if probe.routed_source_type in {
            IngestionSourceType.CSV,
            IngestionSourceType.DELIMITED_TEXT,
        }:
            from app.ingestion.csv_loader import CSVLoader

            dataframe, source_details = CSVLoader().load_raw_dataframe(routed_spec, row_limit=row_limit)
        elif probe.routed_source_type == IngestionSourceType.EXCEL:
            from app.ingestion.excel_loader import ExcelLoader

            dataframe, source_details = ExcelLoader().load_raw_dataframe(routed_spec, row_limit=row_limit)
        elif probe.routed_source_type == IngestionSourceType.HTML_TABLE:
            from app.ingestion.html_table_loader import HTMLTableLoader

            dataframe, source_details = HTMLTableLoader().load_raw_dataframe(routed_spec, row_limit=row_limit)
        else:
            raise UnsupportedSourceError(
                f"Unsupported URL source routing result: {probe.routed_source_type.value}"
            )

        source_details.update(
            {
                "probe_content_type": probe.content_type,
                "probe_final_url": probe.final_url,
                "probe_method": probe.probe_method,
                "routed_source_type": probe.routed_source_type.value,
            }
        )
        return dataframe, source_details


def fetch_url_bytes(url: str, timeout: float = 30.0) -> tuple[bytes, str | None, str]:
    """Fetch a remote resource and return bytes, content-type, and final URL."""

    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
        content_type = _normalize_content_type(response.headers.get("content-type"))
        return response.content, content_type, str(response.url)
    except httpx.HTTPError as exc:
        raise RemoteAccessError(f"Failed to fetch remote dataset: {exc}") from exc


def fetch_url_text(url: str, *, encoding: str = "utf-8", timeout: float = 30.0) -> tuple[str, str | None, str]:
    """Fetch a remote resource and return decoded text, content-type, and final URL."""

    payload, content_type, final_url = fetch_url_bytes(url, timeout=timeout)
    try:
        return payload.decode(encoding), content_type, final_url
    except UnicodeDecodeError as exc:
        raise RemoteAccessError(
            f"Failed to decode remote content using encoding '{encoding}'."
        ) from exc


def probe_url(url: str, timeout: float = 15.0) -> URLProbeResult:
    """Inspect a URL and infer the correct ingestion source type."""

    parsed = urlparse(url)
    extension = Path(parsed.path).suffix.lower()
    query = parse_qs(parsed.query)

    content_type: str | None = None
    final_url = url
    probe_method = "head"

    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            head_response = client.head(url)
            if head_response.status_code >= 400:
                head_response.raise_for_status()
            content_type = _normalize_content_type(head_response.headers.get("content-type"))
            final_url = str(head_response.url)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in {405, 501}:
            raise RemoteAccessError(f"Failed to inspect remote dataset URL: {exc}") from exc
        probe_method = "get-sniff"
    except httpx.HTTPError:
        probe_method = "get-sniff"

    inferred = _infer_source_type(extension=extension, query=query, content_type=content_type)
    if inferred is not None:
        return URLProbeResult(
            final_url=final_url,
            content_type=inferred.content_type,
            routed_source_type=inferred.routed_source_type,
            delimiter=inferred.delimiter,
            probe_method=probe_method,
        )

    sample_bytes, fallback_content_type, final_url = _fetch_sniff_sample(url, timeout=timeout)
    if content_type is None:
        content_type = fallback_content_type

    inferred = _infer_source_type(
        extension=extension,
        query=query,
        content_type=content_type,
        sample_bytes=sample_bytes,
    )
    if inferred is not None:
        return URLProbeResult(
            final_url=final_url,
            content_type=content_type,
            routed_source_type=inferred.routed_source_type,
            delimiter=inferred.delimiter,
            probe_method="get-sniff",
        )

    raise UnsupportedSourceError(
        "Could not determine whether the URL points to a delimited file, Excel file, or HTML table page. "
        "Provide an explicit source_type override instead of using url_file."
    )


def _infer_source_type(
    *,
    extension: str,
    query: dict[str, list[str]],
    content_type: str | None,
    sample_bytes: bytes | None = None,
) -> URLProbeResult | None:
    if content_type in _EXCEL_CONTENT_TYPES or extension in EXCEL_FILE_SUFFIXES:
        return URLProbeResult(
            final_url="",
            content_type=content_type,
            routed_source_type=IngestionSourceType.EXCEL,
        )

    if content_type in _HTML_CONTENT_TYPES or extension in HTML_FILE_SUFFIXES:
        return URLProbeResult(
            final_url="",
            content_type=content_type,
            routed_source_type=IngestionSourceType.HTML_TABLE,
        )

    if content_type == "text/tab-separated-values":
        return URLProbeResult(
            final_url="",
            content_type=content_type,
            routed_source_type=IngestionSourceType.DELIMITED_TEXT,
            delimiter="\t",
        )

    if content_type in _CSV_CONTENT_TYPES or query.get("format") == ["csv"] or extension == ".csv":
        return URLProbeResult(
            final_url="",
            content_type=content_type,
            routed_source_type=IngestionSourceType.CSV,
            delimiter=",",
        )

    if extension == ".tsv":
        return URLProbeResult(
            final_url="",
            content_type=content_type,
            routed_source_type=IngestionSourceType.DELIMITED_TEXT,
            delimiter="\t",
        )

    if content_type in _TEXT_CONTENT_TYPES or extension in DELIMITED_FILE_SUFFIXES:
        from app.ingestion.utils import sniff_delimiter

        delimiter = sniff_delimiter(sample_bytes.decode("utf-8", errors="ignore")) if sample_bytes else None
        if delimiter:
            return URLProbeResult(
                final_url="",
                content_type=content_type,
                routed_source_type=(
                    IngestionSourceType.CSV if delimiter == "," else IngestionSourceType.DELIMITED_TEXT
                ),
                delimiter=delimiter,
            )

    if sample_bytes:
        sample_text = sample_bytes.decode("utf-8", errors="ignore").lower()
        if "<table" in sample_text or "<html" in sample_text:
            return URLProbeResult(
                final_url="",
                content_type=content_type,
                routed_source_type=IngestionSourceType.HTML_TABLE,
            )

    return None


def _fetch_sniff_sample(url: str, timeout: float = 15.0) -> tuple[bytes, str | None, str]:
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                sample = bytearray()
                for chunk in response.iter_bytes():
                    sample.extend(chunk)
                    if len(sample) >= 4096:
                        break
                content_type = _normalize_content_type(response.headers.get("content-type"))
                return bytes(sample), content_type, str(response.url)
    except httpx.HTTPError as exc:
        raise RemoteAccessError(f"Failed to inspect remote dataset URL: {exc}") from exc


def _normalize_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    return content_type.split(";", maxsplit=1)[0].strip().lower()



