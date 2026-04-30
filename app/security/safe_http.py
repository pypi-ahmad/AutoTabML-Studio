"""Bounded, SSRF-resistant HTTP fetch utilities.

Used by ingestion code paths that retrieve user-supplied URLs. Hardened against
the SSRF / unbounded-download / redirect-abuse weaknesses called out in OWASP
Top 10 (A10:2021 - SSRF) and CWE-918 / CWE-400.

Guarantees:

* Only ``http``/``https`` schemes are accepted.
* The hostname is resolved via ``socket.getaddrinfo`` and every returned IP is
  checked against blocked ranges (loopback, link-local, private, multicast,
  reserved). Addresses inside those ranges are rejected before any TCP
  connection is opened.
* Redirects are *not* followed automatically. They are followed manually with a
  configurable hop cap (default 3). Each hop URL is re-validated.
* Responses are streamed; the implementation aborts the download as soon as
  ``Content-Length`` or the running byte total exceeds the configured limit.
* Per-attempt timeouts and a small retry budget for transient failures.
* Suspicious response headers / content types are rejected.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import urlparse, urlsplit

import httpx

logger = logging.getLogger(__name__)


class UnsafeURLError(ValueError):
    """Raised when a URL is rejected by the SSRF / scheme guard."""


class ResponseTooLargeError(RuntimeError):
    """Raised when a remote response exceeds the configured byte cap."""


class UnsafeContentTypeError(RuntimeError):
    """Raised when a response declares a disallowed Content-Type."""


# Defaults are deliberately conservative.
DEFAULT_MAX_BYTES = 200 * 1024 * 1024  # 200 MiB
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_REDIRECTS = 3
DEFAULT_MAX_RETRIES = 2  # number of *retries* (so total attempts = retries + 1)
DEFAULT_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Header names whose presence in a response is treated as suspicious (server
# or upstream proxy is asking the client to do something we don't want, or is
# leaking internal-only signalling).
_SUSPICIOUS_RESPONSE_HEADERS = frozenset(
    {
        "proxy-authenticate",
        "proxy-authorization",
    }
)

# A tight allowlist of content types we ever ingest. ``None`` means "accept
# anything"; callers in ingestion should restrict to this set.
TABULAR_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "text/csv",
        "application/csv",
        "text/plain",
        "text/tab-separated-values",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel.sheet.binary.macroenabled.12",
        "application/octet-stream",
        "text/html",
        "application/xhtml+xml",
        "application/json",
    }
)


@dataclass(frozen=True)
class SafeFetchPolicy:
    """Tunable knobs for ``safe_fetch``."""

    max_bytes: int = DEFAULT_MAX_BYTES
    timeout: float = DEFAULT_TIMEOUT
    max_redirects: int = DEFAULT_MAX_REDIRECTS
    max_retries: int = DEFAULT_MAX_RETRIES
    allowed_schemes: frozenset[str] = field(default_factory=lambda: DEFAULT_ALLOWED_SCHEMES)
    allowed_content_types: frozenset[str] | None = None

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_redirects < 0:
            raise ValueError("max_redirects must be >= 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")


@dataclass
class SafeFetchResult:
    """Successful response payload from :func:`safe_fetch`."""

    content: bytes
    content_type: str | None
    final_url: str
    status_code: int
    headers: Mapping[str, str]


@dataclass
class SafeDownloadResult:
    """Successful streamed download metadata from :func:`safe_download_to_path`."""

    content_type: str | None
    final_url: str
    status_code: int
    headers: Mapping[str, str]
    bytes_written: int


def _validate_url(url: str, allowed_schemes: Iterable[str]) -> tuple[str, str, int]:
    """Validate scheme/host/port, return the parsed (scheme, host, port)."""

    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("URL must be a non-empty string.")
    parsed = urlsplit(url.strip())
    scheme = (parsed.scheme or "").lower()
    if scheme not in {s.lower() for s in allowed_schemes}:
        raise UnsafeURLError(
            f"URL scheme '{scheme or '(empty)'}' is not allowed. Use one of: "
            f"{sorted(allowed_schemes)}."
        )
    if parsed.username or parsed.password:
        raise UnsafeURLError("URLs with embedded credentials are not allowed.")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL is missing a hostname.")
    # Strip IPv6 brackets — already handled by urlsplit, but be defensive.
    host = host.strip("[]")
    try:
        port = parsed.port if parsed.port is not None else (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise UnsafeURLError(f"URL has an invalid port: {exc}") from exc
    if not (0 < port < 65536):
        raise UnsafeURLError(f"URL port {port} is out of range.")
    return scheme, host, port


def _is_blocked_ip(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Return a human-readable reason if ``addr`` is in a blocked range."""

    if addr.is_loopback:
        return "loopback"
    if addr.is_link_local:
        return "link-local"
    if addr.is_private:
        return "private (RFC1918 / RFC4193)"
    if addr.is_multicast:
        return "multicast"
    if addr.is_reserved:
        return "reserved"
    if addr.is_unspecified:
        return "unspecified"
    # IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) — re-evaluate the embedded v4.
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return _is_blocked_ip(addr.ipv4_mapped)
    return None


def _resolve_and_check_host(host: str, port: int) -> list[str]:
    """Resolve a hostname to IPs and reject if any IP is in a blocked range.

    Returns the list of resolved IP strings (informational).
    """

    # If the literal is already an IP, validate it directly and skip DNS.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        reason = _is_blocked_ip(literal)
        if reason is not None:
            raise UnsafeURLError(
                f"Refusing to connect to {host}: address is in blocked range ({reason})."
            )
        return [str(literal)]

    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Could not resolve hostname '{host}': {exc}") from exc

    resolved: list[str] = []
    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        reason = _is_blocked_ip(addr)
        if reason is not None:
            raise UnsafeURLError(
                f"Refusing to connect to {host}: resolved address {ip_str} is in "
                f"blocked range ({reason})."
            )
        resolved.append(ip_str)

    if not resolved:
        raise UnsafeURLError(f"Hostname '{host}' did not resolve to any usable IP address.")
    return resolved


def _normalize_content_type(value: str | None) -> str | None:
    if value is None:
        return None
    return value.split(";", maxsplit=1)[0].strip().lower() or None


def _check_response_headers(
    headers: Mapping[str, str],
    allowed_content_types: frozenset[str] | None,
) -> str | None:
    """Return the normalized content-type after enforcing allowlists."""

    lowered = {k.lower(): v for k, v in headers.items()}
    for suspicious in _SUSPICIOUS_RESPONSE_HEADERS:
        if suspicious in lowered:
            raise UnsafeContentTypeError(
                f"Response carries disallowed header '{suspicious}'."
            )

    content_type = _normalize_content_type(lowered.get("content-type"))
    if allowed_content_types is not None and content_type is not None:
        if content_type not in allowed_content_types:
            raise UnsafeContentTypeError(
                f"Response Content-Type '{content_type}' is not in the allowlist."
            )
    return content_type


def _check_advertised_size(headers: Mapping[str, str], max_bytes: int) -> None:
    raw = headers.get("content-length") or headers.get("Content-Length")
    if not raw:
        return
    try:
        length = int(raw)
    except (TypeError, ValueError):
        return
    if length < 0:
        raise UnsafeContentTypeError("Response advertised a negative Content-Length.")
    if length > max_bytes:
        raise ResponseTooLargeError(
            f"Response advertises {length} bytes which exceeds the cap of {max_bytes} bytes."
        )


def _attempt_fetch(
    url: str,
    *,
    method: str,
    policy: SafeFetchPolicy,
    extra_headers: Mapping[str, str] | None,
) -> SafeFetchResult:
    """Single fetch attempt: validates, follows redirects manually, streams body."""

    current_url = url
    visited: list[str] = []

    timeout = httpx.Timeout(policy.timeout)
    # Disable env-driven proxies by default to avoid surprise outbound paths.
    with httpx.Client(follow_redirects=False, timeout=timeout, trust_env=False) as client:
        for hop in range(policy.max_redirects + 1):
            scheme, host, port = _validate_url(current_url, policy.allowed_schemes)
            _resolve_and_check_host(host, port)
            visited.append(current_url)

            request_headers: dict[str, str] = {"Accept-Encoding": "identity"}
            if extra_headers:
                request_headers.update(extra_headers)

            try:
                with client.stream(method, current_url, headers=request_headers) as response:
                    if 300 <= response.status_code < 400 and "location" in {
                        k.lower() for k in response.headers.keys()
                    }:
                        if hop >= policy.max_redirects:
                            raise UnsafeURLError(
                                f"Exceeded redirect cap ({policy.max_redirects}) starting from {url}."
                            )
                        # Resolve the next hop relative to the *current* URL.
                        next_url = str(response.headers.get("location", "")).strip()
                        if not next_url:
                            raise UnsafeURLError("Redirect response missing Location header.")
                        current_url = _resolve_redirect(current_url, next_url)
                        if current_url in visited:
                            raise UnsafeURLError(f"Redirect loop detected: revisited {current_url}.")
                        # Exiting the inner ``with`` closes the stream.
                        continue

                    response.raise_for_status()
                    content_type = _check_response_headers(response.headers, policy.allowed_content_types)
                    _check_advertised_size(response.headers, policy.max_bytes)

                    buffer = bytearray()
                    if method.upper() == "HEAD":
                        # HEAD responses must not have a body; do not iterate.
                        return SafeFetchResult(
                            content=b"",
                            content_type=content_type,
                            final_url=str(response.url),
                            status_code=response.status_code,
                            headers=dict(response.headers),
                        )
                    for chunk in response.iter_bytes():
                        buffer.extend(chunk)
                        if len(buffer) > policy.max_bytes:
                            raise ResponseTooLargeError(
                                f"Response exceeded the cap of {policy.max_bytes} bytes."
                            )
                    return SafeFetchResult(
                        content=bytes(buffer),
                        content_type=content_type,
                        final_url=str(response.url),
                        status_code=response.status_code,
                        headers=dict(response.headers),
                    )
            except httpx.HTTPStatusError:
                raise
            except httpx.HTTPError:
                raise

    raise UnsafeURLError(f"Failed to obtain a final response for {url}.")  # pragma: no cover - defensive


def _resolve_redirect(current_url: str, next_url: str) -> str:
    """Resolve an absolute-or-relative redirect target."""

    parsed = urlparse(next_url)
    if parsed.scheme and parsed.netloc:
        return next_url
    base = urlparse(current_url)
    # Relative redirect: rebuild against the base.
    if next_url.startswith("//"):
        return f"{base.scheme}:{next_url}"
    if next_url.startswith("/"):
        return f"{base.scheme}://{base.netloc}{next_url}"
    # Path-relative — keep this simple; do not support obscure cases.
    raise UnsafeURLError(f"Refusing to follow non-absolute redirect target '{next_url}'.")


def safe_fetch(
    url: str,
    *,
    method: str = "GET",
    policy: SafeFetchPolicy | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> SafeFetchResult:
    """Fetch ``url`` using the SSRF-resistant, bounded HTTP client.

    Retries are limited to transient httpx transport errors; SSRF/policy errors
    are *never* retried.
    """

    effective_policy = policy or SafeFetchPolicy()
    last_error: Exception | None = None
    for attempt in range(effective_policy.max_retries + 1):
        try:
            return _attempt_fetch(
                url, method=method, policy=effective_policy, extra_headers=extra_headers
            )
        except (UnsafeURLError, ResponseTooLargeError, UnsafeContentTypeError):
            # Policy violations: never retry, propagate immediately.
            raise
        except httpx.HTTPStatusError as exc:
            # Don't retry 4xx (client error). Retry 5xx within the budget.
            if 500 <= exc.response.status_code < 600 and attempt < effective_policy.max_retries:
                last_error = exc
                continue
            raise
        except httpx.HTTPError as exc:
            if attempt < effective_policy.max_retries:
                last_error = exc
                continue
            raise

    # Defensive fallback; loop above always returns or raises.
    raise last_error if last_error else RuntimeError(  # pragma: no cover
        "safe_fetch exhausted retries without an error"
    )


def safe_fetch_text(
    url: str,
    *,
    encoding: str = "utf-8",
    policy: SafeFetchPolicy | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> tuple[str, str | None, str]:
    """Convenience wrapper returning decoded text."""

    result = safe_fetch(url, policy=policy, extra_headers=extra_headers)
    try:
        return result.content.decode(encoding), result.content_type, result.final_url
    except UnicodeDecodeError as exc:
        raise UnsafeContentTypeError(
            f"Failed to decode remote content using encoding '{encoding}'."
        ) from exc


def safe_stream_sample(
    url: str,
    *,
    sample_size: int = 4096,
    policy: SafeFetchPolicy | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> tuple[bytes, str | None, str]:
    """Fetch up to ``sample_size`` bytes for content sniffing.

    Implemented on top of :func:`safe_fetch` but with a dedicated max-bytes cap
    matching ``sample_size`` so even a confused server cannot push more than the
    sample budget at us.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    base = policy or SafeFetchPolicy()
    sample_policy = SafeFetchPolicy(
        max_bytes=sample_size,
        timeout=base.timeout,
        max_redirects=base.max_redirects,
        max_retries=base.max_retries,
        allowed_schemes=base.allowed_schemes,
        allowed_content_types=base.allowed_content_types,
    )
    try:
        result = safe_fetch(url, policy=sample_policy, extra_headers=extra_headers)
    except ResponseTooLargeError:
        # Body was larger than our sample cap; that's expected — re-fetch with a
        # short range request to bound payload while still surfacing a sample.
        result = safe_fetch(
            url,
            policy=sample_policy,
            extra_headers={**(extra_headers or {}), "Range": f"bytes=0-{sample_size - 1}"},
        )
    return result.content[:sample_size], result.content_type, result.final_url


def _attempt_download_to_path(
    url: str,
    *,
    destination_path: Path,
    policy: SafeFetchPolicy,
    extra_headers: Mapping[str, str] | None,
) -> SafeDownloadResult:
    """Single streamed download attempt writing directly to ``destination_path``."""

    current_url = url
    visited: list[str] = []

    timeout = httpx.Timeout(policy.timeout)
    with httpx.Client(follow_redirects=False, timeout=timeout, trust_env=False) as client:
        for hop in range(policy.max_redirects + 1):
            scheme, host, port = _validate_url(current_url, policy.allowed_schemes)
            _resolve_and_check_host(host, port)
            visited.append(current_url)

            request_headers: dict[str, str] = {"Accept-Encoding": "identity"}
            if extra_headers:
                request_headers.update(extra_headers)

            with client.stream("GET", current_url, headers=request_headers) as response:
                if 300 <= response.status_code < 400 and "location" in {
                    k.lower() for k in response.headers.keys()
                }:
                    if hop >= policy.max_redirects:
                        raise UnsafeURLError(
                            f"Exceeded redirect cap ({policy.max_redirects}) starting from {url}."
                        )
                    next_url = str(response.headers.get("location", "")).strip()
                    if not next_url:
                        raise UnsafeURLError("Redirect response missing Location header.")
                    current_url = _resolve_redirect(current_url, next_url)
                    if current_url in visited:
                        raise UnsafeURLError(f"Redirect loop detected: revisited {current_url}.")
                    continue

                response.raise_for_status()
                content_type = _check_response_headers(response.headers, policy.allowed_content_types)
                _check_advertised_size(response.headers, policy.max_bytes)

                bytes_written = 0
                with destination_path.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        bytes_written += len(chunk)
                        if bytes_written > policy.max_bytes:
                            raise ResponseTooLargeError(
                                f"Response exceeded the cap of {policy.max_bytes} bytes."
                            )
                        handle.write(chunk)

                return SafeDownloadResult(
                    content_type=content_type,
                    final_url=str(response.url),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    bytes_written=bytes_written,
                )

    raise UnsafeURLError(f"Failed to obtain a final response for {url}.")  # pragma: no cover


def safe_download_to_path(
    url: str,
    *,
    destination_path: str | Path,
    policy: SafeFetchPolicy | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> SafeDownloadResult:
    """Stream ``url`` into ``destination_path`` with the same guards as :func:`safe_fetch`."""

    effective_policy = policy or SafeFetchPolicy()
    target_path = Path(destination_path)
    last_error: Exception | None = None
    for attempt in range(effective_policy.max_retries + 1):
        try:
            return _attempt_download_to_path(
                url,
                destination_path=target_path,
                policy=effective_policy,
                extra_headers=extra_headers,
            )
        except (UnsafeURLError, ResponseTooLargeError, UnsafeContentTypeError):
            target_path.unlink(missing_ok=True)
            raise
        except httpx.HTTPStatusError as exc:
            target_path.unlink(missing_ok=True)
            if 500 <= exc.response.status_code < 600 and attempt < effective_policy.max_retries:
                last_error = exc
                continue
            raise
        except httpx.HTTPError as exc:
            target_path.unlink(missing_ok=True)
            if attempt < effective_policy.max_retries:
                last_error = exc
                continue
            raise

    raise last_error if last_error else RuntimeError(  # pragma: no cover
        "safe_download_to_path exhausted retries without an error"
    )


# ---------------------------------------------------------------------------
# Async siblings
#
# These mirror the sync helpers above but use ``httpx.AsyncClient`` so callers
# can run many fetches concurrently. The SSRF / redirect / size / content-type
# guards are identical — they share the same validation helpers.
# ---------------------------------------------------------------------------


async def _attempt_fetch_async(
    url: str,
    *,
    method: str,
    policy: SafeFetchPolicy,
    extra_headers: Mapping[str, str] | None,
) -> SafeFetchResult:
    """Async single-attempt counterpart of :func:`_attempt_fetch`."""

    current_url = url
    visited: list[str] = []

    timeout = httpx.Timeout(policy.timeout)
    async with httpx.AsyncClient(
        follow_redirects=False, timeout=timeout, trust_env=False
    ) as client:
        for hop in range(policy.max_redirects + 1):
            scheme, host, port = _validate_url(current_url, policy.allowed_schemes)
            _resolve_and_check_host(host, port)
            visited.append(current_url)

            request_headers: dict[str, str] = {"Accept-Encoding": "identity"}
            if extra_headers:
                request_headers.update(extra_headers)

            async with client.stream(
                method, current_url, headers=request_headers
            ) as response:
                if 300 <= response.status_code < 400 and "location" in {
                    k.lower() for k in response.headers.keys()
                }:
                    if hop >= policy.max_redirects:
                        raise UnsafeURLError(
                            f"Exceeded redirect cap ({policy.max_redirects}) starting from {url}."
                        )
                    next_url = str(response.headers.get("location", "")).strip()
                    if not next_url:
                        raise UnsafeURLError("Redirect response missing Location header.")
                    current_url = _resolve_redirect(current_url, next_url)
                    if current_url in visited:
                        raise UnsafeURLError(
                            f"Redirect loop detected: revisited {current_url}."
                        )
                    continue

                response.raise_for_status()
                content_type = _check_response_headers(
                    response.headers, policy.allowed_content_types
                )
                _check_advertised_size(response.headers, policy.max_bytes)

                if method.upper() == "HEAD":
                    return SafeFetchResult(
                        content=b"",
                        content_type=content_type,
                        final_url=str(response.url),
                        status_code=response.status_code,
                        headers=dict(response.headers),
                    )

                buffer = bytearray()
                async for chunk in response.aiter_bytes():
                    buffer.extend(chunk)
                    if len(buffer) > policy.max_bytes:
                        raise ResponseTooLargeError(
                            f"Response exceeded the cap of {policy.max_bytes} bytes."
                        )
                return SafeFetchResult(
                    content=bytes(buffer),
                    content_type=content_type,
                    final_url=str(response.url),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )

    raise UnsafeURLError(  # pragma: no cover - defensive
        f"Failed to obtain a final response for {url}."
    )


async def safe_fetch_async(
    url: str,
    *,
    method: str = "GET",
    policy: SafeFetchPolicy | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> SafeFetchResult:
    """Async counterpart of :func:`safe_fetch` with identical retry semantics."""

    effective_policy = policy or SafeFetchPolicy()
    last_error: Exception | None = None
    for attempt in range(effective_policy.max_retries + 1):
        try:
            return await _attempt_fetch_async(
                url, method=method, policy=effective_policy, extra_headers=extra_headers
            )
        except (UnsafeURLError, ResponseTooLargeError, UnsafeContentTypeError):
            raise
        except httpx.HTTPStatusError as exc:
            if (
                500 <= exc.response.status_code < 600
                and attempt < effective_policy.max_retries
            ):
                last_error = exc
                continue
            raise
        except httpx.HTTPError as exc:
            if attempt < effective_policy.max_retries:
                last_error = exc
                continue
            raise

    raise last_error if last_error else RuntimeError(  # pragma: no cover
        "safe_fetch_async exhausted retries without an error"
    )


async def safe_fetch_text_async(
    url: str,
    *,
    encoding: str = "utf-8",
    policy: SafeFetchPolicy | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> tuple[str, str | None, str]:
    """Async convenience wrapper returning decoded text."""

    result = await safe_fetch_async(url, policy=policy, extra_headers=extra_headers)
    try:
        return result.content.decode(encoding), result.content_type, result.final_url
    except UnicodeDecodeError as exc:
        raise UnsafeContentTypeError(
            f"Failed to decode remote content using encoding '{encoding}'."
        ) from exc


async def safe_stream_sample_async(
    url: str,
    *,
    sample_size: int = 4096,
    policy: SafeFetchPolicy | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> tuple[bytes, str | None, str]:
    """Async counterpart of :func:`safe_stream_sample`."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    base = policy or SafeFetchPolicy()
    sample_policy = SafeFetchPolicy(
        max_bytes=sample_size,
        timeout=base.timeout,
        max_redirects=base.max_redirects,
        max_retries=base.max_retries,
        allowed_schemes=base.allowed_schemes,
        allowed_content_types=base.allowed_content_types,
    )
    try:
        result = await safe_fetch_async(
            url,
            policy=sample_policy,
            extra_headers=extra_headers,
        )
    except ResponseTooLargeError:
        result = await safe_fetch_async(
            url,
            policy=sample_policy,
            extra_headers={**(extra_headers or {}), "Range": f"bytes=0-{sample_size - 1}"},
        )
    return result.content[:sample_size], result.content_type, result.final_url


async def safe_fetch_many_async(
    urls: Iterable[str],
    *,
    method: str = "GET",
    policy: SafeFetchPolicy | None = None,
    extra_headers: Mapping[str, str] | None = None,
    concurrency: int = 8,
    return_exceptions: bool = True,
) -> list[SafeFetchResult | BaseException]:
    """Fetch many URLs concurrently with bounded parallelism.

    Per-URL guards (SSRF, size, content-type) and retry policy are identical to
    :func:`safe_fetch_async`. ``concurrency`` caps simultaneous requests so the
    local resolver and remote hosts are not hammered.
    """

    # Local import to avoid a circular reference at module load time.
    from app.concurrency import gather_with_concurrency

    coros = [
        safe_fetch_async(
            url, method=method, policy=policy, extra_headers=extra_headers
        )
        for url in urls
    ]
    return await gather_with_concurrency(
        coros, limit=concurrency, return_exceptions=return_exceptions
    )


async def _attempt_download_to_path_async(
    url: str,
    *,
    destination_path: Path,
    policy: SafeFetchPolicy,
    extra_headers: Mapping[str, str] | None,
) -> SafeDownloadResult:
    """Async single-attempt counterpart of :func:`_attempt_download_to_path`."""

    current_url = url
    visited: list[str] = []

    timeout = httpx.Timeout(policy.timeout)
    async with httpx.AsyncClient(
        follow_redirects=False, timeout=timeout, trust_env=False
    ) as client:
        for hop in range(policy.max_redirects + 1):
            scheme, host, port = _validate_url(current_url, policy.allowed_schemes)
            _resolve_and_check_host(host, port)
            visited.append(current_url)

            request_headers: dict[str, str] = {"Accept-Encoding": "identity"}
            if extra_headers:
                request_headers.update(extra_headers)

            async with client.stream(
                "GET", current_url, headers=request_headers
            ) as response:
                if 300 <= response.status_code < 400 and "location" in {
                    k.lower() for k in response.headers.keys()
                }:
                    if hop >= policy.max_redirects:
                        raise UnsafeURLError(
                            f"Exceeded redirect cap ({policy.max_redirects}) starting from {url}."
                        )
                    next_url = str(response.headers.get("location", "")).strip()
                    if not next_url:
                        raise UnsafeURLError("Redirect response missing Location header.")
                    current_url = _resolve_redirect(current_url, next_url)
                    if current_url in visited:
                        raise UnsafeURLError(
                            f"Redirect loop detected: revisited {current_url}."
                        )
                    continue

                response.raise_for_status()
                content_type = _check_response_headers(
                    response.headers, policy.allowed_content_types
                )
                _check_advertised_size(response.headers, policy.max_bytes)

                bytes_written = 0
                # File I/O is local and small per chunk; doing it synchronously
                # inside the async loop keeps the implementation simple without
                # adding aiofiles as a hard dependency.
                with destination_path.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        bytes_written += len(chunk)
                        if bytes_written > policy.max_bytes:
                            raise ResponseTooLargeError(
                                f"Response exceeded the cap of {policy.max_bytes} bytes."
                            )
                        handle.write(chunk)

                return SafeDownloadResult(
                    content_type=content_type,
                    final_url=str(response.url),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    bytes_written=bytes_written,
                )

    raise UnsafeURLError(  # pragma: no cover - defensive
        f"Failed to obtain a final response for {url}."
    )


async def safe_download_to_path_async(
    url: str,
    *,
    destination_path: str | Path,
    policy: SafeFetchPolicy | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> SafeDownloadResult:
    """Async counterpart of :func:`safe_download_to_path`."""

    effective_policy = policy or SafeFetchPolicy()
    target_path = Path(destination_path)
    last_error: Exception | None = None
    for attempt in range(effective_policy.max_retries + 1):
        try:
            return await _attempt_download_to_path_async(
                url,
                destination_path=target_path,
                policy=effective_policy,
                extra_headers=extra_headers,
            )
        except (UnsafeURLError, ResponseTooLargeError, UnsafeContentTypeError):
            target_path.unlink(missing_ok=True)
            raise
        except httpx.HTTPStatusError as exc:
            target_path.unlink(missing_ok=True)
            if (
                500 <= exc.response.status_code < 600
                and attempt < effective_policy.max_retries
            ):
                last_error = exc
                continue
            raise
        except httpx.HTTPError as exc:
            target_path.unlink(missing_ok=True)
            if attempt < effective_policy.max_retries:
                last_error = exc
                continue
            raise

    raise last_error if last_error else RuntimeError(  # pragma: no cover
        "safe_download_to_path_async exhausted retries without an error"
    )
