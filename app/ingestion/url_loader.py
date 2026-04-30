"""URL routing and remote fetch helpers for ingestion."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile
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
from app.security.safe_http import (
    ResponseTooLargeError,
    SafeFetchPolicy,
    TABULAR_CONTENT_TYPES,
    UnsafeContentTypeError,
    UnsafeURLError,
    safe_download_to_path,
    safe_download_to_path_async,
    safe_fetch,
    safe_fetch_async,
    safe_stream_sample,
    safe_stream_sample_async,
)

_DEFAULT_SAMPLE_BYTES = 8192

_HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}
_CSV_CONTENT_TYPES = {"text/csv", "application/csv"}
_TEXT_CONTENT_TYPES = {"text/plain"}
_EXCEL_CONTENT_TYPES = {
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel.sheet.binary.macroenabled.12",
}


def _ingestion_policy(
    timeout: float,
    *,
    max_download_bytes: int | None = None,
    max_redirects: int | None = None,
    max_retries: int | None = None,
) -> SafeFetchPolicy:
    policy_kwargs: dict[str, Any] = {
        "timeout": timeout,
        "allowed_content_types": TABULAR_CONTENT_TYPES,
    }
    if max_download_bytes is not None:
        policy_kwargs["max_bytes"] = max_download_bytes
    if max_redirects is not None:
        policy_kwargs["max_redirects"] = max_redirects
    if max_retries is not None:
        policy_kwargs["max_retries"] = max_retries
    return SafeFetchPolicy(**policy_kwargs)


def _policy_from_input_spec(input_spec: DatasetInputSpec, *, default_timeout: float) -> SafeFetchPolicy:
    timeout = input_spec.url_timeout_seconds or default_timeout
    return _ingestion_policy(
        timeout,
        max_download_bytes=input_spec.url_max_download_bytes,
        max_redirects=input_spec.url_max_redirects,
        max_retries=input_spec.url_max_retries,
    )


@dataclass
class URLProbeResult:
    """Inference result for a remote URL source."""

    final_url: str
    content_type: str | None
    routed_source_type: IngestionSourceType
    delimiter: str | None = None
    probe_method: str = "head"


@dataclass
class DownloadedURLFile:
    """Metadata for a remote resource streamed into a temporary local file."""

    path: Path
    content_type: str | None
    final_url: str
    file_size_bytes: int


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

        probe = probe_url(
            input_spec.url,
            timeout=input_spec.url_timeout_seconds or 15.0,
            max_download_bytes=input_spec.url_max_download_bytes,
            max_redirects=input_spec.url_max_redirects,
            max_retries=input_spec.url_max_retries,
        )
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

    async def load_raw_dataframe_async(
        self,
        input_spec: DatasetInputSpec,
        *,
        row_limit: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not input_spec.url:
            raise UnsupportedSourceError("URL file ingestion requires a URL.")

        probe = await probe_url_async(
            input_spec.url,
            timeout=input_spec.url_timeout_seconds or 15.0,
            max_download_bytes=input_spec.url_max_download_bytes,
            max_redirects=input_spec.url_max_redirects,
            max_retries=input_spec.url_max_retries,
        )
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

            dataframe, source_details = await CSVLoader().load_raw_dataframe_async(
                routed_spec,
                row_limit=row_limit,
            )
        elif probe.routed_source_type == IngestionSourceType.EXCEL:
            from app.ingestion.excel_loader import ExcelLoader

            dataframe, source_details = await ExcelLoader().load_raw_dataframe_async(
                routed_spec,
                row_limit=row_limit,
            )
        elif probe.routed_source_type == IngestionSourceType.HTML_TABLE:
            from app.ingestion.html_table_loader import HTMLTableLoader

            dataframe, source_details = await HTMLTableLoader().load_raw_dataframe_async(
                routed_spec,
                row_limit=row_limit,
            )
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


def fetch_url_bytes(
    url: str,
    *,
    timeout: float = 30.0,
    max_download_bytes: int | None = None,
    max_redirects: int | None = None,
    max_retries: int | None = None,
) -> tuple[bytes, str | None, str]:
    """Fetch a remote resource and return bytes, content-type, and final URL."""

    try:
        result = safe_fetch(
            url,
            policy=_ingestion_policy(
                timeout,
                max_download_bytes=max_download_bytes,
                max_redirects=max_redirects,
                max_retries=max_retries,
            ),
        )
    except (UnsafeURLError, UnsafeContentTypeError, ResponseTooLargeError) as exc:
        raise RemoteAccessError(f"Refused to fetch remote dataset: {exc}") from exc
    except httpx.HTTPError as exc:
        raise RemoteAccessError(f"Failed to fetch remote dataset: {exc}") from exc
    return result.content, result.content_type, result.final_url


@contextmanager
def fetch_url_to_temp_file(
    url: str,
    *,
    timeout: float = 30.0,
    max_download_bytes: int | None = None,
    max_redirects: int | None = None,
    max_retries: int | None = None,
):
    """Stream a remote resource into a temporary file and clean it up afterward."""

    suffix = Path(urlparse(url).path).suffix.lower()
    with tempfile.NamedTemporaryFile(
        prefix="autotabml-ingest-",
        suffix=suffix,
        delete=False,
    ) as temp_handle:
        temp_path = Path(temp_handle.name)

    try:
        result = safe_download_to_path(
            url,
            destination_path=temp_path,
            policy=_ingestion_policy(
                timeout,
                max_download_bytes=max_download_bytes,
                max_redirects=max_redirects,
                max_retries=max_retries,
            ),
        )
    except (UnsafeURLError, UnsafeContentTypeError, ResponseTooLargeError) as exc:
        temp_path.unlink(missing_ok=True)
        raise RemoteAccessError(f"Refused to fetch remote dataset: {exc}") from exc
    except httpx.HTTPError as exc:
        temp_path.unlink(missing_ok=True)
        raise RemoteAccessError(f"Failed to fetch remote dataset: {exc}") from exc

    try:
        yield DownloadedURLFile(
            path=temp_path,
            content_type=result.content_type,
            final_url=result.final_url,
            file_size_bytes=result.bytes_written,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def ensure_local_file_size_guard(
    path: Path,
    *,
    max_bytes: int | None,
) -> int:
    """Reject local files that exceed the configured size guard."""

    file_size = path.stat().st_size
    if max_bytes is not None and file_size > max_bytes:
        raise UnsupportedSourceError(
            f"Local dataset '{path}' is {file_size} bytes which exceeds the configured "
            f"safe read cap of {max_bytes} bytes."
        )
    return file_size


def fetch_url_text(
    url: str,
    *,
    encoding: str = "utf-8",
    timeout: float = 30.0,
    max_download_bytes: int | None = None,
    max_redirects: int | None = None,
    max_retries: int | None = None,
) -> tuple[str, str | None, str]:
    """Fetch a remote resource and return decoded text, content-type, and final URL."""

    payload, content_type, final_url = fetch_url_bytes(
        url,
        timeout=timeout,
        max_download_bytes=max_download_bytes,
        max_redirects=max_redirects,
        max_retries=max_retries,
    )
    try:
        return payload.decode(encoding), content_type, final_url
    except UnicodeDecodeError as exc:
        raise RemoteAccessError(
            f"Failed to decode remote content using encoding '{encoding}'."
        ) from exc


def probe_url(
    url: str,
    *,
    timeout: float = 15.0,
    max_download_bytes: int | None = None,
    max_redirects: int | None = None,
    max_retries: int | None = None,
) -> URLProbeResult:
    """Inspect a URL and infer the correct ingestion source type."""

    parsed = urlparse(url)
    extension = Path(parsed.path).suffix.lower()
    query = parse_qs(parsed.query)

    content_type: str | None = None
    final_url = url
    probe_method = "head"
    policy = _ingestion_policy(
        timeout,
        max_download_bytes=max_download_bytes,
        max_redirects=max_redirects,
        max_retries=max_retries,
    )

    try:
        head_result = safe_fetch(url, method="HEAD", policy=policy)
        content_type = head_result.content_type
        final_url = head_result.final_url
    except (UnsafeURLError, UnsafeContentTypeError, ResponseTooLargeError) as exc:
        raise RemoteAccessError(f"Refused to inspect remote dataset URL: {exc}") from exc
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

    sample_bytes, fallback_content_type, final_url = _fetch_sniff_sample(
        url,
        timeout=timeout,
        max_download_bytes=max_download_bytes,
        max_redirects=max_redirects,
        max_retries=max_retries,
    )
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


async def probe_url_async(
    url: str,
    *,
    timeout: float = 15.0,
    max_download_bytes: int | None = None,
    max_redirects: int | None = None,
    max_retries: int | None = None,
) -> URLProbeResult:
    """Async counterpart of :func:`probe_url`."""

    parsed = urlparse(url)
    extension = Path(parsed.path).suffix.lower()
    query = parse_qs(parsed.query)

    content_type: str | None = None
    final_url = url
    probe_method = "head"
    policy = _ingestion_policy(
        timeout,
        max_download_bytes=max_download_bytes,
        max_redirects=max_redirects,
        max_retries=max_retries,
    )

    try:
        head_result = await safe_fetch_async(url, method="HEAD", policy=policy)
        content_type = head_result.content_type
        final_url = head_result.final_url
    except (UnsafeURLError, UnsafeContentTypeError, ResponseTooLargeError) as exc:
        raise RemoteAccessError(f"Refused to inspect remote dataset URL: {exc}") from exc
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

    sample_bytes, fallback_content_type, final_url = await _fetch_sniff_sample_async(
        url,
        timeout=timeout,
        max_download_bytes=max_download_bytes,
        max_redirects=max_redirects,
        max_retries=max_retries,
    )
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


def _fetch_sniff_sample(
    url: str,
    *,
    timeout: float = 15.0,
    max_download_bytes: int | None = None,
    max_redirects: int | None = None,
    max_retries: int | None = None,
) -> tuple[bytes, str | None, str]:
    sample_size = 4096 if max_download_bytes is None else min(4096, max_download_bytes)
    try:
        return safe_stream_sample(
            url,
            sample_size=sample_size,
            policy=_ingestion_policy(
                timeout,
                max_download_bytes=max_download_bytes,
                max_redirects=max_redirects,
                max_retries=max_retries,
            ),
        )
    except (UnsafeURLError, UnsafeContentTypeError, ResponseTooLargeError) as exc:
        raise RemoteAccessError(f"Refused to inspect remote dataset URL: {exc}") from exc
    except httpx.HTTPError as exc:
        raise RemoteAccessError(f"Failed to inspect remote dataset URL: {exc}") from exc


async def _fetch_sniff_sample_async(
    url: str,
    *,
    timeout: float = 15.0,
    max_download_bytes: int | None = None,
    max_redirects: int | None = None,
    max_retries: int | None = None,
) -> tuple[bytes, str | None, str]:
    sample_size = 4096 if max_download_bytes is None else min(4096, max_download_bytes)
    try:
        return await safe_stream_sample_async(
            url,
            sample_size=sample_size,
            policy=_ingestion_policy(
                timeout,
                max_download_bytes=max_download_bytes,
                max_redirects=max_redirects,
                max_retries=max_retries,
            ),
        )
    except (UnsafeURLError, UnsafeContentTypeError, ResponseTooLargeError) as exc:
        raise RemoteAccessError(f"Refused to inspect remote dataset URL: {exc}") from exc
    except httpx.HTTPError as exc:
        raise RemoteAccessError(f"Failed to inspect remote dataset URL: {exc}") from exc


def _normalize_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    return content_type.split(";", maxsplit=1)[0].strip().lower()


# ---------------------------------------------------------------------------
# Async fetch helpers
# ---------------------------------------------------------------------------


async def async_fetch_url_bytes(
    url: str,
    *,
    timeout: float = 30.0,
    max_download_bytes: int | None = None,
    max_redirects: int | None = None,
    max_retries: int | None = None,
) -> tuple[bytes, str | None, str]:
    """Async counterpart of :func:`fetch_url_bytes`."""

    try:
        result = await safe_fetch_async(
            url,
            policy=_ingestion_policy(
                timeout,
                max_download_bytes=max_download_bytes,
                max_redirects=max_redirects,
                max_retries=max_retries,
            ),
        )
    except (UnsafeURLError, UnsafeContentTypeError, ResponseTooLargeError) as exc:
        raise RemoteAccessError(f"Refused to fetch remote dataset: {exc}") from exc
    except httpx.HTTPError as exc:
        raise RemoteAccessError(f"Failed to fetch remote dataset: {exc}") from exc
    return result.content, result.content_type, result.final_url


@asynccontextmanager
async def async_fetch_url_to_temp_file(
    url: str,
    *,
    timeout: float = 30.0,
    max_download_bytes: int | None = None,
    max_redirects: int | None = None,
    max_retries: int | None = None,
):
    """Async counterpart of :func:`fetch_url_to_temp_file`."""

    suffix = Path(urlparse(url).path).suffix.lower()
    with tempfile.NamedTemporaryFile(
        prefix="autotabml-ingest-",
        suffix=suffix,
        delete=False,
    ) as temp_handle:
        temp_path = Path(temp_handle.name)

    try:
        result = await safe_download_to_path_async(
            url,
            destination_path=temp_path,
            policy=_ingestion_policy(
                timeout,
                max_download_bytes=max_download_bytes,
                max_redirects=max_redirects,
                max_retries=max_retries,
            ),
        )
    except (UnsafeURLError, UnsafeContentTypeError, ResponseTooLargeError) as exc:
        temp_path.unlink(missing_ok=True)
        raise RemoteAccessError(f"Refused to fetch remote dataset: {exc}") from exc
    except httpx.HTTPError as exc:
        temp_path.unlink(missing_ok=True)
        raise RemoteAccessError(f"Failed to fetch remote dataset: {exc}") from exc

    try:
        yield DownloadedURLFile(
            path=temp_path,
            content_type=result.content_type,
            final_url=result.final_url,
            file_size_bytes=result.bytes_written,
        )
    finally:
        temp_path.unlink(missing_ok=True)



