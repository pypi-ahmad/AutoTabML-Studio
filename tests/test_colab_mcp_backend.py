"""Tests for Colab MCP backend and notebook infrastructure."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.backends import build_backend
from app.backends.colab_mcp_backend import ColabMCPExecutionBackend
from app.backends.local_backend import LocalExecutionBackend
from app.config.enums import ExecutionBackend
from app.config.models import AppSettings, ExecutionSettings

# ---------------------------------------------------------------------------
# Default backend is Colab MCP
# ---------------------------------------------------------------------------

class TestDefaultBackendIsColabMCP:
    def test_execution_settings_defaults_to_colab_mcp(self):
        settings = ExecutionSettings()
        assert settings.backend == ExecutionBackend.COLAB_MCP

    def test_app_settings_defaults_to_colab_mcp(self):
        settings = AppSettings()
        assert settings.execution.backend == ExecutionBackend.COLAB_MCP

    def test_can_override_to_local(self):
        settings = AppSettings.model_validate({"execution": {"backend": "local"}})
        assert settings.execution.backend == ExecutionBackend.LOCAL


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

class TestBuildBackend:
    def test_builds_colab_mcp_backend(self):
        backend = build_backend(ExecutionBackend.COLAB_MCP)
        assert isinstance(backend, ColabMCPExecutionBackend)
        assert backend.backend_type == ExecutionBackend.COLAB_MCP

    def test_builds_local_backend(self):
        backend = build_backend(ExecutionBackend.LOCAL)
        assert isinstance(backend, LocalExecutionBackend)
        assert backend.backend_type == ExecutionBackend.LOCAL

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown execution backend"):
            build_backend("nonexistent")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ColabMCPExecutionBackend
# ---------------------------------------------------------------------------

class TestColabMCPExecutionBackend:
    def test_initial_state(self):
        backend = ColabMCPExecutionBackend()
        assert backend.is_connected is False
        assert backend.backend_type == ExecutionBackend.COLAB_MCP

    @pytest.mark.asyncio
    async def test_validate_backend_fails_without_uvx(self, monkeypatch):
        monkeypatch.setattr(
            "app.backends.colab_mcp_backend._find_uvx", lambda: None
        )
        backend = ColabMCPExecutionBackend()
        assert await backend.validate_backend() is False

    @pytest.mark.asyncio
    async def test_validate_backend_fails_without_mcp_sdk(self, monkeypatch):
        monkeypatch.setattr(
            "app.backends.colab_mcp_backend._find_uvx", lambda: "/usr/bin/uvx"
        )
        # Simulate mcp not importable
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "mcp":
                raise ImportError("no mcp")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        backend = ColabMCPExecutionBackend()
        assert await backend.validate_backend() is False

    @pytest.mark.asyncio
    async def test_validate_backend_ok_when_both_present(self, monkeypatch):
        monkeypatch.setattr(
            "app.backends.colab_mcp_backend._find_uvx", lambda: "/usr/bin/uvx"
        )
        # Ensure the mcp import inside validate_backend succeeds regardless of
        # whether the real mcp package is installed in this test environment.
        fake_mcp = type(sys)("mcp")
        fake_mcp.ClientSession = object
        monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
        backend = ColabMCPExecutionBackend()
        assert await backend.validate_backend() is True

    @pytest.mark.asyncio
    async def test_prepare_session_error_without_uvx(self, monkeypatch):
        monkeypatch.setattr(
            "app.backends.colab_mcp_backend._find_uvx", lambda: None
        )
        # Ensure the mcp import inside prepare_session succeeds so we reach the
        # uvx-not-found branch rather than the mcp-missing branch.
        fake_mcp = type(sys)("mcp")
        fake_mcp.ClientSession = object
        fake_mcp.StdioServerParameters = object
        fake_mcp_client = type(sys)("mcp.client")
        fake_mcp_stdio = type(sys)("mcp.client.stdio")
        fake_mcp_stdio.stdio_client = object
        monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
        monkeypatch.setitem(sys.modules, "mcp.client", fake_mcp_client)
        monkeypatch.setitem(sys.modules, "mcp.client.stdio", fake_mcp_stdio)
        backend = ColabMCPExecutionBackend()
        result = await backend.prepare_session()
        assert result["status"] == "error"
        assert "uvx" in result["detail"]

    @pytest.mark.asyncio
    async def test_run_job_raises_when_not_connected(self):
        backend = ColabMCPExecutionBackend()
        with pytest.raises(RuntimeError, match="not connected"):
            await backend.run_job({"tool": "some_tool", "arguments": {}})

    @pytest.mark.asyncio
    async def test_run_job_raises_without_tool_key(self):
        backend = ColabMCPExecutionBackend()
        backend._connected = True
        backend._session = MagicMock()
        with pytest.raises(ValueError, match="'tool' key"):
            await backend.run_job({"arguments": {}})

    @pytest.mark.asyncio
    async def test_run_job_calls_session_tool(self):
        backend = ColabMCPExecutionBackend()
        backend._connected = True

        mock_content = MagicMock()
        mock_content.text = "Hello from Colab"
        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_result.isError = False

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        backend._session = mock_session

        result = await backend.run_job({
            "tool": "execute_cell",
            "arguments": {"code": "print('hi')"},
        })
        assert result["success"] is True
        assert result["output"] == "Hello from Colab"
        mock_session.call_tool.assert_awaited_once_with(
            "execute_cell", {"code": "print('hi')"}
        )

    @pytest.mark.asyncio
    async def test_list_tools_empty_when_no_session(self):
        backend = ColabMCPExecutionBackend()
        assert await backend.list_tools() == []

    @pytest.mark.asyncio
    async def test_list_tools_queries_session(self):
        backend = ColabMCPExecutionBackend()

        mock_tool = MagicMock()
        mock_tool.name = "execute_cell"
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)
        backend._session = mock_session

        tools = await backend.list_tools()
        assert tools == ["execute_cell"]

    @pytest.mark.asyncio
    async def test_cleanup_resets_state(self):
        backend = ColabMCPExecutionBackend()
        backend._connected = True
        backend._available_tools = ["tool1"]
        backend._session = MagicMock()
        backend._exit_stack = None

        await backend.cleanup()
        assert backend.is_connected is False
        assert backend._available_tools == []
        assert backend._session is None


# ---------------------------------------------------------------------------
# LocalExecutionBackend still works
# ---------------------------------------------------------------------------

class TestLocalExecutionBackend:
    @pytest.mark.asyncio
    async def test_validate_backend_always_true(self):
        backend = LocalExecutionBackend()
        assert await backend.validate_backend() is True

    @pytest.mark.asyncio
    async def test_prepare_session_returns_ready(self):
        backend = LocalExecutionBackend()
        result = await backend.prepare_session()
        assert result["status"] == "ready"


# ---------------------------------------------------------------------------
# Startup diagnostics for Colab MCP
# ---------------------------------------------------------------------------

class TestStartupColabMCPDiagnostics:
    def test_startup_warns_when_uvx_missing(self, tmp_path: Path, monkeypatch):
        import shutil as _shutil

        monkeypatch.setattr(_shutil, "which", lambda _name: None)
        settings = AppSettings.model_validate({
            "artifacts": {"root_dir": str(tmp_path / "runtime")},
            "execution": {"backend": "colab_mcp"},
        })
        from app.startup import initialize_local_runtime
        status = initialize_local_runtime(settings, include_optional_network_checks=False)
        messages = [i.message for i in status.issues]
        assert any("uvx" in m for m in messages)

    def test_startup_warns_when_mcp_missing(self, tmp_path: Path, monkeypatch):
        import shutil as _shutil

        monkeypatch.setattr(_shutil, "which", lambda _name: "/usr/bin/uvx")
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "mcp":
                raise ImportError("no mcp")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        settings = AppSettings.model_validate({
            "artifacts": {"root_dir": str(tmp_path / "runtime")},
            "execution": {"backend": "colab_mcp"},
        })
        from app.startup import initialize_local_runtime
        status = initialize_local_runtime(settings, include_optional_network_checks=False)
        messages = [i.message for i in status.issues]
        assert any("mcp" in m for m in messages)

    def test_startup_no_colab_warnings_for_local_backend(self, tmp_path: Path):
        settings = AppSettings.model_validate({
            "artifacts": {"root_dir": str(tmp_path / "runtime")},
            "execution": {"backend": "local"},
        })
        from app.startup import initialize_local_runtime
        status = initialize_local_runtime(settings, include_optional_network_checks=False)
        messages = [i.message for i in status.issues]
        assert not any("uvx" in m or "Colab MCP" in m for m in messages)


# ---------------------------------------------------------------------------
# Packaging: colab extra
# ---------------------------------------------------------------------------

class TestPackagingColabExtra:
    def test_colab_extra_includes_mcp(self):
        import tomllib
        data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        colab_deps = data["project"]["optional-dependencies"]["colab"]
        assert any("mcp" in dep for dep in colab_deps)


# ---------------------------------------------------------------------------
# Enum ordering: Colab MCP first
# ---------------------------------------------------------------------------

class TestEnumOrdering:
    def test_colab_mcp_is_first_backend(self):
        backends = list(ExecutionBackend)
        assert backends[0] == ExecutionBackend.COLAB_MCP
        assert backends[1] == ExecutionBackend.LOCAL
