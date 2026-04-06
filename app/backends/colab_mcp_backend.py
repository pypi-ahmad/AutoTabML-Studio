"""Colab MCP execution backend – connects to Google Colab via MCP.

This backend spawns the ``colab-mcp`` server (via ``uvx``) as a subprocess,
communicates over the standard MCP stdio transport, and proxies notebook
operations (add cell, execute code, etc.) to the user's open Colab session.

Prerequisites:
- ``uv`` must be installed (``pip install uv``)
- A Google Colab notebook must be open in the user's browser
- The ``colab-mcp`` package is fetched automatically by ``uvx``

Ollama is intentionally unsupported for this backend (local-only provider).
"""

from __future__ import annotations

import logging
import shutil
from contextlib import AsyncExitStack
from typing import Any

from app.backends.base import BaseExecutionBackend
from app.config.enums import ExecutionBackend

logger = logging.getLogger(__name__)

# Tool names exposed by the Colab MCP server once a browser session is connected.
_TOOL_OPEN_CONNECTION = "open_colab_browser_connection"
_COLAB_MCP_PACKAGE = "git+https://github.com/googlecolab/colab-mcp"
_UVX_TIMEOUT_SECS = 30


def _find_uvx() -> str | None:
    """Return the path to ``uvx`` if it is installed, else *None*."""
    return shutil.which("uvx")


class ColabMCPExecutionBackend(BaseExecutionBackend):
    """Execution backend that delegates work to a Google Colab runtime via MCP."""

    backend_type = ExecutionBackend.COLAB_MCP

    def __init__(self) -> None:
        self._session: Any | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._connected: bool = False
        self._available_tools: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def validate_backend(self) -> bool:
        """Check that ``uvx`` is installed and the MCP SDK is importable."""
        if _find_uvx() is None:
            logger.warning(
                "uvx is not installed. Install it with: pip install uv"
            )
            return False
        try:
            from mcp import ClientSession  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "mcp SDK is not installed. Install it with: pip install 'mcp>=1.0'"
            )
            return False

    async def prepare_session(self) -> dict[str, Any]:
        """Spawn the Colab MCP server and establish a client session.

        Returns a dict with connection metadata. If the server cannot be
        reached the ``status`` key will be ``"error"``.
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            return {
                "backend": "colab_mcp",
                "status": "error",
                "detail": "mcp SDK is not installed",
            }

        uvx_path = _find_uvx()
        if not uvx_path:
            return {
                "backend": "colab_mcp",
                "status": "error",
                "detail": "uvx not found on PATH",
            }

        server_params = StdioServerParameters(
            command=uvx_path,
            args=[_COLAB_MCP_PACKAGE],
        )

        self._exit_stack = AsyncExitStack()
        try:
            transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = transport
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
            tools_result = await self._session.list_tools()
            self._available_tools = [t.name for t in tools_result.tools]
            self._connected = True
            logger.info(
                "Colab MCP session ready – %d tools available: %s",
                len(self._available_tools),
                self._available_tools,
            )
            return {
                "backend": "colab_mcp",
                "status": "ready",
                "tools": self._available_tools,
            }
        except Exception as exc:
            logger.error("Failed to start Colab MCP session: %s", exc)
            await self.cleanup()
            return {
                "backend": "colab_mcp",
                "status": "error",
                "detail": str(exc),
            }

    async def run_job(self, job_payload: dict[str, Any]) -> dict[str, Any]:
        """Execute a job by calling Colab MCP tools.

        ``job_payload`` must include a ``"tool"`` key with the MCP tool name
        and an ``"arguments"`` dict forwarded to the tool.
        """
        if not self._connected or self._session is None:
            raise RuntimeError(
                "Colab MCP session is not connected. Call prepare_session() first."
            )
        tool_name: str = job_payload.get("tool", "")
        arguments: dict[str, Any] = job_payload.get("arguments", {})
        if not tool_name:
            raise ValueError("job_payload must include a 'tool' key")

        result = await self._session.call_tool(tool_name, arguments)
        output_parts: list[str] = []
        for content in result.content:
            if hasattr(content, "text"):
                output_parts.append(content.text)
        return {
            "tool": tool_name,
            "success": not result.isError,
            "output": "\n".join(output_parts),
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    async def open_browser_connection(self) -> dict[str, Any]:
        """Call ``open_colab_browser_connection`` to link to a Colab notebook."""
        return await self.run_job({
            "tool": _TOOL_OPEN_CONNECTION,
            "arguments": {},
        })

    async def list_tools(self) -> list[str]:
        """Return the currently available MCP tool names."""
        if self._session is None:
            return []
        tools_result = await self._session.list_tools()
        self._available_tools = [t.name for t in tools_result.tools]
        return self._available_tools

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def cleanup(self) -> None:
        """Tear down the MCP session and server subprocess."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as exc:
                logger.debug("Colab MCP cleanup error (ignored): %s", exc)
            self._exit_stack = None
        self._session = None
        self._connected = False
        self._available_tools = []
