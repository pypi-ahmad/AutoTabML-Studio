"""YData Profiling runner – wraps ydata-profiling behind a clean interface.

All ydata-profiling imports are contained here so the rest of the app
never depends on the library directly.
"""

from __future__ import annotations

import contextlib
import io
import logging
import shutil
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from app.artifacts import ArtifactKind, LocalArtifactManager
from app.config.models import ProfilingMode
from app.profiling.base import BaseProfilingService
from app.profiling.errors import ProfilingError, ProfilingSetupError
from app.profiling.schemas import ProfilingArtifactBundle, ProfilingConfig, ProfilingResultSummary
from app.profiling.selectors import maybe_sample, select_profiling_mode
from app.profiling.summary import extract_summary

logger = logging.getLogger(__name__)

_YDATA_AVAILABLE: bool | None = None
_YDATA_IMPORT_ERROR: ImportError | None = None


@contextlib.contextmanager
def _suppress_profiling_runtime_noise():
    """Suppress known non-actionable third-party warnings during profiling."""

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            from pyparsing.warnings import PyparsingDeprecationWarning
        except Exception:  # pragma: no cover - optional dependency surface
            PyparsingDeprecationWarning = None

        if PyparsingDeprecationWarning is not None:
            warnings.filterwarnings("ignore", category=PyparsingDeprecationWarning)
        warnings.filterwarnings(
            "ignore",
            message=r"datetime\.datetime\.utcnow\(\) is deprecated.*",
            category=DeprecationWarning,
        )
        yield


def profiling_install_guidance(import_error: ImportError | None = None) -> str:
    """Return a user-facing installation hint for profiling dependencies."""
    effective_error = import_error or _YDATA_IMPORT_ERROR
    install_command = 'pip install "ydata-profiling" "setuptools<82"'

    if effective_error is not None:
        missing_name = getattr(effective_error, "name", None)
        if missing_name == "pkg_resources" or "pkg_resources" in str(effective_error):
            return (
                "Data profiling is not available yet. "
                "A required system package is incompatible. "
                "Ask your administrator to reinstall the profiling add-on."
            )

    return (
        "Data profiling is not available yet. "
        "The profiling add-on has not been installed. "
        "Ask your administrator to set it up."
    )


def is_ydata_available() -> bool:
    """Return True if ydata-profiling is importable."""
    global _YDATA_AVAILABLE, _YDATA_IMPORT_ERROR
    if _YDATA_AVAILABLE is None:
        try:
            with _suppress_profiling_runtime_noise():
                import ydata_profiling  # noqa: F401
            _YDATA_AVAILABLE = True
            _YDATA_IMPORT_ERROR = None
        except ImportError as exc:
            _YDATA_AVAILABLE = False
            _YDATA_IMPORT_ERROR = exc
    return _YDATA_AVAILABLE


class YDataProfilingService(BaseProfilingService):
    """Profiling service backed by ydata-profiling."""

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self._artifacts_dir = artifacts_dir

    def profile(
        self,
        df: pd.DataFrame,
        config: ProfilingConfig,
        *,
        dataset_name: str | None = None,
    ) -> tuple[ProfilingResultSummary, ProfilingArtifactBundle | None]:
        if not is_ydata_available():
            raise ProfilingSetupError(profiling_install_guidance())

        effective_mode = select_profiling_mode(df, config)
        working_df, was_sampled, sample_size_used = maybe_sample(df, config)

        report = self._generate_report(working_df, effective_mode, config.title or dataset_name)
        summary = extract_summary(
            report,
            df,
            effective_mode=effective_mode,
            was_sampled=was_sampled,
            sample_size_used=sample_size_used,
            dataset_name=dataset_name,
        )

        bundle: ProfilingArtifactBundle | None = None
        if self._artifacts_dir is not None:
            bundle = self._write_artifacts(report, summary, dataset_name)

        return summary, bundle

    def _generate_report(
        self,
        df: pd.DataFrame,
        mode: ProfilingMode,
        title: str | None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if mode == ProfilingMode.MINIMAL:
            kwargs["minimal"] = True
        if title:
            kwargs["title"] = title
        kwargs["progress_bar"] = False

        try:
            with _suppress_profiling_runtime_noise():
                from ydata_profiling import ProfileReport  # noqa: WPS433
                return ProfileReport(df, **kwargs)
        except Exception as exc:
            raise ProfilingError(f"ydata-profiling report generation failed: {exc}") from exc

    def _write_artifacts(
        self,
        report: Any,
        summary: ProfilingResultSummary,
        dataset_name: str | None,
    ) -> ProfilingArtifactBundle:
        artifacts_dir = self._artifacts_dir
        assert artifacts_dir is not None
        manager = LocalArtifactManager()

        bundle = ProfilingArtifactBundle()

        html_path = manager.build_artifact_path(
            kind=ArtifactKind.PROFILING,
            stem=dataset_name,
            label="profile",
            suffix=".html",
            timestamp=summary.run_timestamp,
            output_dir=artifacts_dir,
        )
        try:
            temporary_html_path = manager.create_temp_file_path(stem=dataset_name, suffix=".html")
            with _suppress_profiling_runtime_noise():
                report.to_file(str(temporary_html_path))
            shutil.move(str(temporary_html_path), str(html_path))
            bundle.html_report_path = html_path
            logger.info("Profiling HTML report written to %s", html_path)
        except Exception as exc:
            logger.warning("Failed to write profiling HTML report: %s", exc)

        json_path = manager.build_artifact_path(
            kind=ArtifactKind.PROFILING,
            stem=dataset_name,
            label="profile_summary",
            suffix=".json",
            timestamp=summary.run_timestamp,
            output_dir=artifacts_dir,
        )
        manager.write_text(json_path, summary.model_dump_json(indent=2))
        bundle.summary_json_path = json_path
        logger.info("Profiling summary JSON written to %s", json_path)

        return bundle
