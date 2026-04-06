"""Integration test — real Colab MCP server handshake (no mocks).

Marked ``integration`` so it only runs in CI's integration job or when
explicitly selected (``pytest -m integration``).  Skips automatically if
``uvx`` or the ``mcp`` SDK is absent.
"""

from __future__ import annotations

import shutil

import pytest

# Skip entire module if prerequisites are missing.
uvx_path = shutil.which("uvx")
try:
    from mcp import ClientSession  # noqa: F401

    _mcp_ok = True
except ImportError:
    _mcp_ok = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(uvx_path is None, reason="uvx not on PATH"),
    pytest.mark.skipif(not _mcp_ok, reason="mcp SDK not installed"),
]


from app.backends.colab_mcp_backend import ColabMCPExecutionBackend  # noqa: E402


class TestColabMCPRealHandshake:
    """Prove the real colab-mcp server spawns and the MCP handshake works."""

    async def test_validate_backend_returns_true(self):
        backend = ColabMCPExecutionBackend()
        assert await backend.validate_backend() is True

    async def test_prepare_session_connects_and_lists_tools(self):
        backend = ColabMCPExecutionBackend()
        try:
            result = await backend.prepare_session()

            assert result["backend"] == "colab_mcp"
            assert result["status"] == "ready", f"Unexpected status: {result}"

            # The server must expose at least open_colab_browser_connection
            tools = result["tools"]
            assert isinstance(tools, list)
            assert len(tools) >= 1
            assert "open_colab_browser_connection" in tools
        finally:
            await backend.cleanup()

    async def test_list_tools_after_connect(self):
        backend = ColabMCPExecutionBackend()
        try:
            result = await backend.prepare_session()
            assert result["status"] == "ready"

            tools = await backend.list_tools()
            assert "open_colab_browser_connection" in tools
        finally:
            await backend.cleanup()

    async def test_cleanup_resets_connection(self):
        backend = ColabMCPExecutionBackend()
        result = await backend.prepare_session()
        assert result["status"] == "ready"
        assert backend.is_connected is True

        await backend.cleanup()
        assert backend.is_connected is False
        assert backend._session is None
        assert backend._available_tools == []

    async def test_run_job_without_browser_returns_error(self):
        """Calling a tool that needs a browser session should fail gracefully."""
        backend = ColabMCPExecutionBackend()
        try:
            result = await backend.prepare_session()
            assert result["status"] == "ready"

            # open_colab_browser_connection should return an error or
            # non-success when no browser is actually connected.
            job_result = await backend.run_job({
                "tool": "open_colab_browser_connection",
                "arguments": {},
            })
            # We just verify structurally valid response — no crash.
            assert "tool" in job_result
            assert "output" in job_result
            assert isinstance(job_result["success"], bool)
        finally:
            await backend.cleanup()
