"""Artifact generation for FLAML AutoML runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.artifacts import ArtifactKind, LocalArtifactManager
from app.modeling.flaml.schemas import FlamlArtifactBundle, FlamlResultBundle


def write_flaml_artifacts(
    bundle: FlamlResultBundle,
    artifacts_dir: Path,
) -> FlamlArtifactBundle:
    """Write FLAML artifacts to disk and return their paths."""

    manager = LocalArtifactManager()
    artifact_bundle = FlamlArtifactBundle()

    if bundle.search_result is not None:
        search_json_path = manager.build_artifact_path(
            kind=ArtifactKind.EXPERIMENT,
            stem=bundle.dataset_name,
            label="flaml_search_result",
            suffix=".json",
            timestamp=bundle.summary.run_timestamp,
            output_dir=artifacts_dir,
        )
        manager.write_text(search_json_path, bundle.search_result.model_dump_json(indent=2))
        artifact_bundle.search_result_json_path = search_json_path

        if bundle.search_result.leaderboard:
            leaderboard_df = pd.DataFrame(
                [row.model_dump(mode="json") for row in bundle.search_result.leaderboard]
            )
            leaderboard_csv_path = manager.build_artifact_path(
                kind=ArtifactKind.EXPERIMENT,
                stem=bundle.dataset_name,
                label="flaml_leaderboard",
                suffix=".csv",
                timestamp=bundle.summary.run_timestamp,
                output_dir=artifacts_dir,
            )
            manager.write_dataframe_csv(leaderboard_csv_path, leaderboard_df, index=False)
            artifact_bundle.leaderboard_csv_path = leaderboard_csv_path

            leaderboard_json_path = manager.build_artifact_path(
                kind=ArtifactKind.EXPERIMENT,
                stem=bundle.dataset_name,
                label="flaml_leaderboard",
                suffix=".json",
                timestamp=bundle.summary.run_timestamp,
                output_dir=artifacts_dir,
            )
            manager.write_json(
                leaderboard_json_path,
                [row.model_dump(mode="json") for row in bundle.search_result.leaderboard],
            )
            artifact_bundle.leaderboard_json_path = leaderboard_json_path

    summary_json_path = manager.build_artifact_path(
        kind=ArtifactKind.EXPERIMENT,
        stem=bundle.dataset_name,
        label="flaml_summary",
        suffix=".json",
        timestamp=bundle.summary.run_timestamp,
        output_dir=artifacts_dir,
    )
    manager.write_text(summary_json_path, bundle.summary.model_dump_json(indent=2))
    artifact_bundle.summary_json_path = summary_json_path

    return artifact_bundle
