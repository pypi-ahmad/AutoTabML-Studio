"""Startup checks and local-runtime initialization for AutoTabML Studio."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from app.artifacts import LocalArtifactManager
from app.config.enums import ExecutionBackend, LLMProvider
from app.config.models import AppSettings
from app.storage import AppMetadataStore


class StartupIssue(BaseModel):
    """One startup diagnostic item."""

    severity: Literal["info", "warning", "error"]
    message: str


class StartupStatus(BaseModel):
    """Initialization result for local app runtime resources."""

    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    artifact_dirs: list[Path] = Field(default_factory=list)
    database_path: Path | None = None
    temp_files_removed: int = 0
    partial_files_removed: int = 0
    issues: list[StartupIssue] = Field(default_factory=list)

    @property
    def warnings(self) -> list[StartupIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def errors(self) -> list[StartupIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]


def initialize_local_runtime(
    settings: AppSettings,
    *,
    include_optional_network_checks: bool = False,
) -> StartupStatus:
    """Prepare conservative local runtime resources and collect actionable diagnostics."""

    status = StartupStatus()
    manager = LocalArtifactManager(settings.artifacts)
    try:
        status.artifact_dirs = manager.ensure_directories()
    except Exception as exc:
        status.issues.append(StartupIssue(severity="error", message=f"Could not create artifact directories: {exc}"))
        return status

    try:
        status.temp_files_removed = len(manager.cleanup_stale_temp_artifacts())
        status.partial_files_removed = len(manager.cleanup_failed_partial_artifacts())
    except Exception as exc:
        status.issues.append(StartupIssue(severity="warning", message=f"Artifact cleanup skipped: {exc}"))

    if settings.database.initialize_on_startup:
        try:
            store = AppMetadataStore(settings.database.path)
            store.initialize()
            status.database_path = store.db_path
        except Exception as exc:
            status.issues.append(StartupIssue(severity="error", message=f"Could not initialize local app database: {exc}"))

    status.issues.extend(_validate_mlflow_settings(settings))

    if include_optional_network_checks and settings.provider.provider == LLMProvider.OLLAMA:
        ollama_issue = _validate_ollama_endpoint(settings.ollama_base_url)
        if ollama_issue is not None:
            status.issues.append(ollama_issue)

    if settings.execution.backend == ExecutionBackend.COLAB_MCP:
        status.issues.extend(_validate_colab_mcp_prerequisites())

    return status


def format_startup_issues(status: StartupStatus) -> list[str]:
    """Return concise display strings for startup diagnostics."""

    return [f"[{issue.severity}] {issue.message}" for issue in status.issues]


def _validate_mlflow_settings(settings: AppSettings) -> list[StartupIssue]:
    issues: list[StartupIssue] = []
    for label, value in (("tracking_uri", settings.mlflow.tracking_uri), ("registry_uri", settings.mlflow.registry_uri)):
        if not value:
            continue
        cleaned = value.strip()
        if not cleaned:
            issues.append(StartupIssue(severity="warning", message=f"Configured MLflow {label} is blank and will be ignored."))
            continue
        if any(char.isspace() for char in cleaned):
            issues.append(StartupIssue(severity="warning", message=f"Configured MLflow {label} contains whitespace and may be invalid: {cleaned!r}"))
            continue
        parsed = urlparse(cleaned)
        looks_like_local_path = Path(cleaned).anchor or cleaned.startswith(".") or cleaned.startswith("/")
        if "://" in cleaned and not parsed.scheme:
            issues.append(StartupIssue(severity="warning", message=f"Configured MLflow {label} is malformed: {cleaned!r}"))
        elif not parsed.scheme and not looks_like_local_path:
            issues.append(StartupIssue(severity="warning", message=f"Configured MLflow {label} does not look like a URI or local path: {cleaned!r}"))
    return issues


def _validate_ollama_endpoint(base_url: str) -> StartupIssue | None:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
        if response.status_code == 200:
            return None
        return StartupIssue(
            severity="warning",
            message=(
                f"Ollama is selected but {base_url.rstrip('/')} returned status {response.status_code}. "
                "Start Ollama locally or switch providers in Settings."
            ),
        )
    except httpx.HTTPError as exc:
        return StartupIssue(
            severity="warning",
            message=(
                f"Ollama is selected but is not reachable at {base_url.rstrip('/')} ({exc.__class__.__name__}). "
                "Start Ollama locally or switch providers in Settings."
            ),
        )


def _validate_colab_mcp_prerequisites() -> list[StartupIssue]:
    """Report missing prerequisites for the Colab MCP backend."""
    import shutil

    issues: list[StartupIssue] = []
    if shutil.which("uvx") is None:
        issues.append(
            StartupIssue(
                severity="warning",
                message=(
                    "Colab MCP backend selected but 'uvx' is not on PATH. "
                    "Install it with: pip install uv"
                ),
            )
        )
    try:
        import mcp  # noqa: F401
    except ImportError:
        issues.append(
            StartupIssue(
                severity="warning",
                message=(
                    "Colab MCP backend selected but the 'mcp' SDK is not installed. "
                    "Install it with: pip install 'mcp>=1.0'"
                ),
            )
        )
    return issues