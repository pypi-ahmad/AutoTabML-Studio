"""Centralized local artifact path and lifecycle management.

This manager only handles workspace-local app artifacts. MLflow-managed artifact
stores remain separate and are not routed through this layer.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

import pandas as pd

from app.config.models import ArtifactSettings
from app.path_utils import safe_artifact_stem


class ArtifactKind(str, Enum):
    """Supported local artifact directory kinds."""

    VALIDATION = "validation"
    PROFILING = "profiling"
    BENCHMARK = "benchmark"
    EXPERIMENT = "experiment"
    MODEL = "model"
    COMPARISON = "comparison"
    PREDICTION = "prediction"
    TEMP = "temp"


class LocalArtifactManager:
    """Create, write, and conservatively clean local workspace artifacts."""

    def __init__(self, settings: ArtifactSettings | None = None) -> None:
        self._settings = settings or ArtifactSettings()

    @property
    def settings(self) -> ArtifactSettings:
        return self._settings

    def ensure_directories(self) -> list[Path]:
        paths = [
            self._settings.root_dir,
            self._settings.validation_dir,
            self._settings.profiling_dir,
            self._settings.benchmark_dir,
            self._settings.experiments_dir,
            self._settings.models_dir,
            self._settings.snapshots_dir,
            self._settings.comparisons_dir,
            self._settings.predictions_dir,
            self._settings.temp_dir,
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def directory_for(self, kind: ArtifactKind) -> Path:
        mapping = {
            ArtifactKind.VALIDATION: self._settings.validation_dir,
            ArtifactKind.PROFILING: self._settings.profiling_dir,
            ArtifactKind.BENCHMARK: self._settings.benchmark_dir,
            ArtifactKind.EXPERIMENT: self._settings.experiments_dir,
            ArtifactKind.MODEL: self._settings.models_dir,
            ArtifactKind.COMPARISON: self._settings.comparisons_dir,
            ArtifactKind.PREDICTION: self._settings.predictions_dir,
            ArtifactKind.TEMP: self._settings.temp_dir,
        }
        return mapping[kind]

    def build_artifact_path(
        self,
        *,
        kind: ArtifactKind,
        stem: str | None,
        suffix: str,
        label: str | None = None,
        timestamp: datetime | None = None,
        output_dir: Path | None = None,
        ensure_unique: bool = False,
    ) -> Path:
        base_dir = output_dir or self.directory_for(kind)
        base_dir.mkdir(parents=True, exist_ok=True)

        effective_timestamp = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
        safe_stem = safe_artifact_stem(stem, default=kind.value)
        parts = [safe_stem]
        if label:
            parts.append(safe_artifact_stem(label, default=label))
        parts.append(effective_timestamp.strftime("%Y%m%dT%H%M%S"))
        candidate = base_dir / ("_".join(parts) + suffix)
        return self._ensure_unique(candidate) if ensure_unique else candidate

    def create_temp_file_path(self, *, stem: str | None = None, suffix: str = "") -> Path:
        temp_dir = self.directory_for(ArtifactKind.TEMP)
        temp_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = safe_artifact_stem(stem, default="temp")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        candidate = temp_dir / f"{safe_stem}_{timestamp}_{uuid4().hex[:8]}{suffix}"
        return self._ensure_unique(candidate)

    def write_text(self, path: Path, content: str, *, encoding: str = "utf-8") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = self._partial_path_for(path)
        partial_path.write_text(content, encoding=encoding)
        partial_path.replace(path)
        return path

    def write_bytes(self, path: Path, content: bytes) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = self._partial_path_for(path)
        partial_path.write_bytes(content)
        partial_path.replace(path)
        return path

    def write_json(self, path: Path, payload) -> Path:  # noqa: ANN001
        return self.write_text(path, json.dumps(payload, indent=2, default=str))

    def write_dataframe_csv(self, path: Path, dataframe: pd.DataFrame, *, index: bool = False) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = self._partial_path_for(path)
        dataframe.to_csv(partial_path, index=index)
        partial_path.replace(path)
        return path

    def cleanup_stale_temp_artifacts(self, older_than_hours: int | None = None) -> list[Path]:
        threshold_hours = older_than_hours or self._settings.temp_retention_hours
        return self._cleanup_paths(self.directory_for(ArtifactKind.TEMP), older_than_hours=threshold_hours)

    def cleanup_failed_partial_artifacts(self, older_than_hours: int | None = None) -> list[Path]:
        threshold_hours = older_than_hours or self._settings.failed_partial_retention_hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=threshold_hours)
        removed: list[Path] = []
        for path in self._settings.root_dir.rglob("*.partial"):
            try:
                modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if modified_at < cutoff:
                    path.unlink(missing_ok=True)
                    removed.append(path)
            except OSError:
                continue
        return removed

    def _cleanup_paths(self, directory: Path, *, older_than_hours: int) -> list[Path]:
        if older_than_hours <= 0 or not directory.exists():
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
        removed: list[Path] = []
        for path in directory.iterdir():
            if path.is_dir():
                continue
            try:
                modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if modified_at < cutoff:
                    path.unlink(missing_ok=True)
                    removed.append(path)
            except OSError:
                continue
        return removed

    def _ensure_unique(self, path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        counter = 2
        while True:
            candidate = path.with_name(f"{stem}_{counter}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def _partial_path_for(self, path: Path) -> Path:
        return path.with_name(f"{path.name}.{uuid4().hex}.partial")