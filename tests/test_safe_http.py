"""Security tests for the bounded HTTP fetch utility (SSRF guard)."""

from __future__ import annotations

import socket

import httpx
import pytest
import respx

from app.ingestion.errors import RemoteAccessError
from app.ingestion.url_loader import fetch_url_bytes, probe_url
from app.security.safe_http import (
    ResponseTooLargeError,
    SafeFetchPolicy,
    TABULAR_CONTENT_TYPES,
    UnsafeContentTypeError,
    UnsafeURLError,
    safe_download_to_path,
    safe_fetch,
)


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch):
    """Force hostname resolution to a *public* IP regardless of the host string.

    This lets respx intercept httpx traffic without needing real DNS, and
    sidesteps the SSRF guard for tests that intentionally hit non-blocked
    hosts.
    """

    def _fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", port))]

    monkeypatch.setattr("app.security.safe_http.socket.getaddrinfo", _fake_getaddrinfo)
    return _fake_getaddrinfo


class TestSchemeAndCredentialGuards:
    @pytest.mark.parametrize(
        "url",
        [
            "ftp://example.com/data.csv",
            "file:///etc/passwd",
            "gopher://example.com/data",
            "javascript:alert(1)",
            "data:text/plain,hello",
            "",
            "   ",
        ],
    )
    def test_disallowed_scheme_is_rejected(self, url):
        with pytest.raises(UnsafeURLError):
            safe_fetch(url)

    def test_url_with_credentials_is_rejected(self):
        with pytest.raises(UnsafeURLError, match="credentials"):
            safe_fetch("https://user:pass@example.com/data.csv")

    def test_url_without_hostname_is_rejected(self):
        with pytest.raises(UnsafeURLError, match="hostname"):
            safe_fetch("https:///path")


class TestSSRFHostBlocking:
    @pytest.mark.parametrize(
        "url, label",
        [
            ("http://127.0.0.1/secret", "loopback"),
            ("http://127.255.255.254/secret", "loopback"),
            ("http://[::1]/secret", "loopback"),
            ("http://localhost/secret", "loopback"),
            ("http://169.254.169.254/latest/meta-data/", "link-local"),  # AWS IMDS
            ("http://10.0.0.1/internal", "private"),
            ("http://192.168.1.1/admin", "private"),
            ("http://172.16.0.1/internal", "private"),
            ("http://0.0.0.0/anything", "unspecified"),
            ("http://[fc00::1]/internal", "private"),
            ("http://[fe80::1]/internal", "link-local"),
        ],
    )
    def test_blocked_ip_literal_is_rejected(self, url, label):
        with pytest.raises(UnsafeURLError, match="blocked range"):
            safe_fetch(url)

    def test_dns_to_loopback_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        def _resolve_to_loopback(host, port, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port))]

        monkeypatch.setattr("app.security.safe_http.socket.getaddrinfo", _resolve_to_loopback)
        with pytest.raises(UnsafeURLError, match="blocked range"):
            safe_fetch("https://innocent.example.com/data.csv")

    def test_dns_to_aws_metadata_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        def _resolve_to_imds(host, port, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("169.254.169.254", port))]

        monkeypatch.setattr("app.security.safe_http.socket.getaddrinfo", _resolve_to_imds)
        with pytest.raises(UnsafeURLError, match="blocked range"):
            safe_fetch("https://attacker-controlled.example.com/anything")

    def test_ipv4_mapped_loopback_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        def _resolve_to_mapped_loopback(host, port, *args, **kwargs):
            return [(socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("::ffff:127.0.0.1", port, 0, 0))]

        monkeypatch.setattr("app.security.safe_http.socket.getaddrinfo", _resolve_to_mapped_loopback)
        with pytest.raises(UnsafeURLError, match="blocked range"):
            safe_fetch("https://attacker.example.com/data")


class TestRedirectGuards:
    @respx.mock
    def test_redirects_are_not_followed_automatically(self, public_dns):
        url = "https://example.com/data"
        respx.get(url).mock(
            return_value=httpx.Response(302, headers={"location": "https://example.com/next"})
        )
        respx.get("https://example.com/next").mock(
            return_value=httpx.Response(200, content=b"x,y\n1,2\n", headers={"content-type": "text/csv"})
        )

        result = safe_fetch(url)
        assert result.content == b"x,y\n1,2\n"

    @respx.mock
    def test_redirect_cap_exceeded_is_rejected(self, public_dns):
        # Build a chain longer than the default cap of 3.
        respx.get("https://example.com/0").mock(
            return_value=httpx.Response(302, headers={"location": "https://example.com/1"})
        )
        respx.get("https://example.com/1").mock(
            return_value=httpx.Response(302, headers={"location": "https://example.com/2"})
        )
        respx.get("https://example.com/2").mock(
            return_value=httpx.Response(302, headers={"location": "https://example.com/3"})
        )
        respx.get("https://example.com/3").mock(
            return_value=httpx.Response(302, headers={"location": "https://example.com/4"})
        )

        with pytest.raises(UnsafeURLError, match="redirect cap"):
            safe_fetch("https://example.com/0")

    @respx.mock
    def test_redirect_loop_is_detected(self, public_dns):
        respx.get("https://example.com/a").mock(
            return_value=httpx.Response(302, headers={"location": "https://example.com/b"})
        )
        respx.get("https://example.com/b").mock(
            return_value=httpx.Response(302, headers={"location": "https://example.com/a"})
        )
        with pytest.raises(UnsafeURLError, match="loop"):
            safe_fetch("https://example.com/a")

    @respx.mock
    def test_redirect_to_blocked_host_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        # First hop resolves public; redirect target resolves loopback.
        host_to_addr = {
            "good.example.com": "8.8.8.8",
            "evil.internal": "127.0.0.1",
        }

        def _fake_getaddrinfo(host, port, *args, **kwargs):
            ip = host_to_addr.get(host, "8.8.8.8")
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]

        monkeypatch.setattr("app.security.safe_http.socket.getaddrinfo", _fake_getaddrinfo)
        respx.get("https://good.example.com/start").mock(
            return_value=httpx.Response(302, headers={"location": "https://evil.internal/secret"})
        )

        with pytest.raises(UnsafeURLError, match="blocked range"):
            safe_fetch("https://good.example.com/start")

    @respx.mock
    def test_relative_redirect_is_rejected(self, public_dns):
        respx.get("https://example.com/start").mock(
            return_value=httpx.Response(302, headers={"location": "next-page"})
        )
        with pytest.raises(UnsafeURLError, match="non-absolute"):
            safe_fetch("https://example.com/start")


class TestSizeLimits:
    @respx.mock
    def test_advertised_content_length_over_cap_is_rejected(self, public_dns):
        url = "https://example.com/big"
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=b"x" * 10,
                headers={"content-type": "text/csv", "content-length": "10000000000"},
            )
        )
        policy = SafeFetchPolicy(max_bytes=1024)
        with pytest.raises(ResponseTooLargeError):
            safe_fetch(url, policy=policy)


class TestRetryBehavior:
    @respx.mock
    def test_transient_transport_error_retries_within_limit(self, public_dns):
        url = "https://example.com/retry"
        attempts = {"count": 0}

        def _side_effect(request: httpx.Request):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.ReadTimeout("temporary timeout")
            return httpx.Response(200, content=b"x,y\n1,2\n", headers={"content-type": "text/csv"})

        respx.get(url).mock(side_effect=_side_effect)

        result = safe_fetch(url, policy=SafeFetchPolicy(max_retries=1))

        assert attempts["count"] == 2
        assert result.content == b"x,y\n1,2\n"

    @respx.mock
    def test_transient_transport_error_stops_after_retry_limit(self, public_dns):
        url = "https://example.com/retry-fail"

        def _side_effect(request: httpx.Request):
            raise httpx.ReadTimeout("temporary timeout")

        respx.get(url).mock(side_effect=_side_effect)

        with pytest.raises(httpx.ReadTimeout, match="temporary timeout"):
            safe_fetch(url, policy=SafeFetchPolicy(max_retries=1))

    @respx.mock
    def test_streaming_aborts_when_body_exceeds_cap(self, public_dns):
        url = "https://example.com/big"
        # Don't advertise content-length; deliver a body larger than the cap.
        big_payload = b"x" * 8192
        respx.get(url).mock(
            return_value=httpx.Response(200, content=big_payload, headers={"content-type": "text/csv"})
        )
        policy = SafeFetchPolicy(max_bytes=1024)
        with pytest.raises(ResponseTooLargeError):
            safe_fetch(url, policy=policy)


class TestDownloadToPath:
    @respx.mock
    def test_safe_download_to_path_writes_streamed_body_to_disk(self, public_dns, tmp_path):
        url = "https://example.com/data.csv"
        destination = tmp_path / "downloaded.csv"
        payload = b"x,y\n1,2\n"

        respx.get(url).mock(
            return_value=httpx.Response(200, content=payload, headers={"content-type": "text/csv"})
        )

        result = safe_download_to_path(url, destination_path=destination)

        assert destination.read_bytes() == payload
        assert result.bytes_written == len(payload)
        assert result.content_type == "text/csv"


class TestContentTypeGuards:
    @respx.mock
    def test_disallowed_content_type_is_rejected(self, public_dns):
        url = "https://example.com/binary"
        respx.get(url).mock(
            return_value=httpx.Response(
                200, content=b"\x00", headers={"content-type": "application/x-msdownload"}
            )
        )
        policy = SafeFetchPolicy(allowed_content_types=TABULAR_CONTENT_TYPES)
        with pytest.raises(UnsafeContentTypeError, match="not in the allowlist"):
            safe_fetch(url, policy=policy)

    @respx.mock
    def test_proxy_authenticate_header_is_rejected(self, public_dns):
        url = "https://example.com/data"
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=b"x,y\n1,2\n",
                headers={"content-type": "text/csv", "proxy-authenticate": "Basic"},
            )
        )
        with pytest.raises(UnsafeContentTypeError, match="proxy-authenticate"):
            safe_fetch(url)


class TestIngestionIntegration:
    """Confirm ingestion-facing wrappers translate guard errors into RemoteAccessError."""

    def test_fetch_url_bytes_translates_ssrf_to_remote_access_error(self):
        with pytest.raises(RemoteAccessError, match="Refused"):
            fetch_url_bytes("http://127.0.0.1/secret")

    def test_probe_url_translates_ssrf_to_remote_access_error(self):
        with pytest.raises(RemoteAccessError, match="Refused"):
            probe_url("http://192.168.1.5/data.csv")

    def test_fetch_url_bytes_rejects_unsupported_scheme(self):
        with pytest.raises(RemoteAccessError):
            fetch_url_bytes("ftp://example.com/data.csv")
