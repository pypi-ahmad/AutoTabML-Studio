"""Tests for centralized local artifact lifecycle management."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.artifacts import ArtifactKind, LocalArtifactManager
from app.config.models import AppSettings


def _artifact_settings_for(root_dir: Path):
    settings = AppSettings.model_validate({"artifacts": {"root_dir": str(root_dir)}})
    return settings.artifacts


class TestLocalArtifactManager:
    def test_ensure_directories_and_write_helpers(self, tmp_path: Path):
        artifact_settings = _artifact_settings_for(tmp_path / "artifacts")
        manager = LocalArtifactManager(artifact_settings)

        created_paths = manager.ensure_directories()
        timestamp = datetime(2026, 4, 4, 12, 0, 0, tzinfo=timezone.utc)
        json_path = manager.build_artifact_path(
            kind=ArtifactKind.VALIDATION,
            stem="unsafe:name",
            label="summary",
            suffix=".json",
            timestamp=timestamp,
        )
        csv_path = manager.build_artifact_path(
            kind=ArtifactKind.PREDICTION,
            stem="scored-output",
            suffix=".csv",
            timestamp=timestamp,
        )

        manager.write_json(json_path, {"ok": 1})
        manager.write_dataframe_csv(csv_path, pd.DataFrame({"prediction": ["yes"]}))

        assert all(path.exists() for path in created_paths)
        assert json_path.exists()
        assert csv_path.exists()
        assert "unsafe_name" in json_path.name
        assert csv_path.read_text(encoding="utf-8").startswith("prediction")

    def test_build_artifact_path_can_opt_into_uniqueness(self, tmp_path: Path):
        artifact_settings = _artifact_settings_for(tmp_path / "artifacts")
        manager = LocalArtifactManager(artifact_settings)
        timestamp = datetime(2026, 4, 4, 12, 0, 0, tzinfo=timezone.utc)

        first_path = manager.build_artifact_path(
            kind=ArtifactKind.EXPERIMENT,
            stem="run",
            suffix=".json",
            timestamp=timestamp,
        )
        manager.write_text(first_path, "{}")
        second_path = manager.build_artifact_path(
            kind=ArtifactKind.EXPERIMENT,
            stem="run",
            suffix=".json",
            timestamp=timestamp,
            ensure_unique=True,
        )

        assert first_path != second_path
        assert second_path.stem.endswith("_2")

    def test_cleanup_removes_stale_temp_and_partial_files(self, tmp_path: Path):
        artifact_settings = _artifact_settings_for(tmp_path / "artifacts")
        manager = LocalArtifactManager(artifact_settings)
        manager.ensure_directories()

        stale_temp = artifact_settings.temp_dir / "old.tmp"
        stale_partial = artifact_settings.validation_dir / "result.json.partial"
        fresh_temp = artifact_settings.temp_dir / "fresh.tmp"

        stale_temp.write_text("old", encoding="utf-8")
        stale_partial.write_text("partial", encoding="utf-8")
        fresh_temp.write_text("fresh", encoding="utf-8")

        old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=4)).timestamp()
        os.utime(stale_temp, (old_timestamp, old_timestamp))
        os.utime(stale_partial, (old_timestamp, old_timestamp))

        removed_temp = manager.cleanup_stale_temp_artifacts(older_than_hours=1)
        removed_partial = manager.cleanup_failed_partial_artifacts(older_than_hours=1)

        assert stale_temp in removed_temp
        assert stale_partial in removed_partial
        assert not stale_temp.exists()
        assert not stale_partial.exists()
        assert fresh_temp.exists()