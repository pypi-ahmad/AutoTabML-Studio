"""Artifact generation for FLAML AutoML runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.artifacts import ArtifactKind, LocalArtifactManager
from app.modeling.base import BaseArtifacts
from app.modeling.flaml.schemas import FlamlArtifactBundle, FlamlResultBundle


class FlamlArtifactsWriter(BaseArtifacts[FlamlResultBundle, FlamlArtifactBundle]):
    """Shared-path artifact writer for FLAML result bundles."""

    artifact_kind = ArtifactKind.EXPERIMENT
    artifact_bundle_cls = FlamlArtifactBundle

    def build(self) -> FlamlArtifactBundle:
        if self.bundle.search_result is not None:
            search_json_path = self._artifact_path(label="flaml_search_result", suffix=".json")
            self._write_text(search_json_path, self.bundle.search_result.model_dump_json(indent=2))
            self.artifacts.search_result_json_path = search_json_path

            if self.bundle.search_result.leaderboard:
                leaderboard_df = pd.DataFrame(
                    [row.model_dump(mode="json") for row in self.bundle.search_result.leaderboard]
                )
                leaderboard_csv_path = self._artifact_path(label="flaml_leaderboard", suffix=".csv")
                self._write_dataframe(leaderboard_csv_path, leaderboard_df, index=False)
                self.artifacts.leaderboard_csv_path = leaderboard_csv_path

                leaderboard_json_path = self._artifact_path(label="flaml_leaderboard", suffix=".json")
                self._write_json(
                    leaderboard_json_path,
                    [row.model_dump(mode="json") for row in self.bundle.search_result.leaderboard],
                )
                self.artifacts.leaderboard_json_path = leaderboard_json_path

        summary_json_path = self._artifact_path(label="flaml_summary", suffix=".json")
        self._write_text(summary_json_path, self.bundle.summary.model_dump_json(indent=2))
        self.artifacts.summary_json_path = summary_json_path

        return self.artifacts


def write_flaml_artifacts(
    bundle: FlamlResultBundle,
    artifacts_dir: Path,
) -> FlamlArtifactBundle:
    """Write FLAML artifacts to disk and return their paths."""

    return FlamlArtifactsWriter(bundle, artifacts_dir).build()
