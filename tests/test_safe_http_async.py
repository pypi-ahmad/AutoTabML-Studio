"""Async tests for the async siblings in :mod:`app.security.safe_http`."""

from __future__ import annotations

import socket
from pathlib import Path

import httpx
import pytest
import respx

from app.security.safe_http import (
    ResponseTooLargeError,
    SafeFetchPolicy,
    UnsafeURLError,
    safe_download_to_path_async,
    safe_fetch_async,
    safe_fetch_many_async,
)


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch):
    """Make hostname resolution return a public address so respx can intercept."""

    def _fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", port))
        ]

    monkeypatch.setattr("app.security.safe_http.socket.getaddrinfo", _fake_getaddrinfo)
    return _fake_getaddrinfo


class TestSafeFetchAsync:
    @pytest.mark.asyncio
    @respx.mock
    async def test_fetches_csv_payload(self, public_dns):
        url = "https://example.com/data.csv"
        respx.get(url).mock(
            return_value=httpx.Response(
                200, content=b"a,b\n1,2\n", headers={"content-type": "text/csv"}
            )
        )

        result = await safe_fetch_async(url)

        assert result.content == b"a,b\n1,2\n"
        assert result.content_type == "text/csv"
        assert result.status_code == 200

    @pytest.mark.asyncio
    @respx.mock
    async def test_follows_redirects_manually(self, public_dns):
        respx.get("https://example.com/start").mock(
            return_value=httpx.Response(
                302, headers={"location": "https://example.com/final"}
            )
        )
        respx.get("https://example.com/final").mock(
            return_value=httpx.Response(
                200, content=b"ok", headers={"content-type": "text/csv"}
            )
        )

        result = await safe_fetch_async("https://example.com/start")

        assert result.content == b"ok"
        assert result.final_url.endswith("/final")

    @pytest.mark.asyncio
    @respx.mock
    async def test_size_cap_is_enforced(self, public_dns):
        url = "https://example.com/big.csv"
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=b"x" * 5000,
                headers={"content-type": "text/csv"},
            )
        )

        with pytest.raises(ResponseTooLargeError):
            await safe_fetch_async(url, policy=SafeFetchPolicy(max_bytes=512))

    @pytest.mark.asyncio
    async def test_rejects_blocked_scheme(self):
        with pytest.raises(UnsafeURLError):
            await safe_fetch_async("ftp://example.com/data")

    @pytest.mark.asyncio
    async def test_rejects_loopback(self, monkeypatch: pytest.MonkeyPatch):
        def _resolve_loopback(host, port, *args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port))
            ]

        monkeypatch.setattr("app.security.safe_http.socket.getaddrinfo", _resolve_loopback)

        with pytest.raises(UnsafeURLError, match="blocked range"):
            await safe_fetch_async("https://attacker.example.com/")


class TestSafeFetchManyAsync:
    @pytest.mark.asyncio
    @respx.mock
    async def test_fetches_multiple_urls_concurrently(self, public_dns):
        respx.get("https://example.com/a").mock(
            return_value=httpx.Response(
                200, content=b"AAA", headers={"content-type": "text/csv"}
            )
        )
        respx.get("https://example.com/b").mock(
            return_value=httpx.Response(
                200, content=b"BBB", headers={"content-type": "text/csv"}
            )
        )
        respx.get("https://example.com/c").mock(
            return_value=httpx.Response(
                200, content=b"CCC", headers={"content-type": "text/csv"}
            )
        )

        results = await safe_fetch_many_async(
            [
                "https://example.com/a",
                "https://example.com/b",
                "https://example.com/c",
            ],
            concurrency=3,
        )

        assert [r.content for r in results] == [b"AAA", b"BBB", b"CCC"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_collects_per_url_failures(self, public_dns):
        respx.get("https://example.com/ok").mock(
            return_value=httpx.Response(
                200, content=b"ok", headers={"content-type": "text/csv"}
            )
        )
        respx.get("https://example.com/big").mock(
            return_value=httpx.Response(
                200,
                content=b"x" * 4000,
                headers={"content-type": "text/csv"},
            )
        )

        results = await safe_fetch_many_async(
            ["https://example.com/ok", "https://example.com/big"],
            policy=SafeFetchPolicy(max_bytes=512),
            concurrency=2,
        )

        assert results[0].content == b"ok"
        assert isinstance(results[1], ResponseTooLargeError)


class TestSafeDownloadToPathAsync:
    @pytest.mark.asyncio
    @respx.mock
    async def test_streams_response_to_disk(self, public_dns, tmp_path: Path):
        url = "https://example.com/file.csv"
        respx.get(url).mock(
            return_value=httpx.Response(
                200, content=b"a,b\n1,2\n", headers={"content-type": "text/csv"}
            )
        )

        target = tmp_path / "out.csv"
        result = await safe_download_to_path_async(url, destination_path=target)

        assert target.read_bytes() == b"a,b\n1,2\n"
        assert result.bytes_written == len(b"a,b\n1,2\n")

    @pytest.mark.asyncio
    @respx.mock
    async def test_size_cap_aborts_download(self, public_dns, tmp_path: Path):
        url = "https://example.com/big.csv"
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=b"x" * 4000,
                headers={"content-type": "text/csv"},
            )
        )

        target = tmp_path / "big.csv"
        with pytest.raises(ResponseTooLargeError):
            await safe_download_to_path_async(
                url, destination_path=target, policy=SafeFetchPolicy(max_bytes=512)
            )
        assert not target.exists()
