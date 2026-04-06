"""Tests for startup diagnostics and local runtime initialization."""

from __future__ import annotations

from pathlib import Path

import httpx

from app.config.models import AppSettings
from app.startup import format_startup_issues, initialize_local_runtime


class TestStartupRuntime:
    def test_initialize_local_runtime_creates_resources_and_reports_invalid_mlflow_uri(self, tmp_path: Path):
        settings = AppSettings.model_validate(
            {
                "artifacts": {"root_dir": str(tmp_path / "runtime")},
                "execution": {"backend": "local"},
                "mlflow": {"tracking_uri": "bad uri"},
            }
        )

        status = initialize_local_runtime(settings, include_optional_network_checks=False)

        assert status.database_path == settings.database.path
        assert all(path.exists() for path in status.artifact_dirs)
        assert any("tracking_uri" in issue.message for issue in status.warnings)
        assert any(line.startswith("[warning]") for line in format_startup_issues(status))

    def test_initialize_local_runtime_checks_ollama_when_selected(self, tmp_path: Path, monkeypatch):
        settings = AppSettings.model_validate(
            {
                "artifacts": {"root_dir": str(tmp_path / "runtime")},
                "execution": {"backend": "local"},
                "provider": {"provider": "ollama"},
            }
        )

        def _raise(*args, **kwargs):  # noqa: ANN002, ANN003
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr("app.startup.httpx.get", _raise)

        status = initialize_local_runtime(settings, include_optional_network_checks=True)

        assert any("Ollama is selected" in issue.message for issue in status.warnings)