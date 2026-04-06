"""Artifact generation for experiment runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.artifacts import ArtifactKind, LocalArtifactManager
from app.modeling.pycaret.schemas import ExperimentArtifactBundle, ExperimentResultBundle
from app.modeling.pycaret.summary import leaderboard_to_dataframe


def write_experiment_artifacts(
    bundle: ExperimentResultBundle,
    artifacts_dir: Path,
) -> ExperimentArtifactBundle:
    """Write experiment artifacts to disk and return their paths."""

    manager = LocalArtifactManager()
    artifact_bundle = ExperimentArtifactBundle(plot_artifacts=list(bundle.evaluation_plots))
    if bundle.saved_model_artifacts:
        artifact_bundle.saved_model_metadata_paths = [
            artifact.metadata_path
            for artifact in bundle.saved_model_artifacts
            if artifact.metadata_path is not None
        ]

    setup_json_path = manager.build_artifact_path(
        kind=ArtifactKind.EXPERIMENT,
        stem=bundle.dataset_name,
        label="experiment_setup",
        suffix=".json",
        timestamp=bundle.summary.run_timestamp,
        output_dir=artifacts_dir,
    )
    manager.write_text(setup_json_path, bundle.summary.setup_config.model_dump_json(indent=2))
    artifact_bundle.setup_json_path = setup_json_path

    if bundle.available_metrics:
        metrics_df = pd.DataFrame([row.model_dump(mode="json") for row in bundle.available_metrics])
        metrics_csv_path = manager.build_artifact_path(
            kind=ArtifactKind.EXPERIMENT,
            stem=bundle.dataset_name,
            label="experiment_metrics",
            suffix=".csv",
            timestamp=bundle.summary.run_timestamp,
            output_dir=artifacts_dir,
        )
        manager.write_dataframe_csv(metrics_csv_path, metrics_df, index=False)
        artifact_bundle.metrics_csv_path = metrics_csv_path

        metrics_json_path = manager.build_artifact_path(
            kind=ArtifactKind.EXPERIMENT,
            stem=bundle.dataset_name,
            label="experiment_metrics",
            suffix=".json",
            timestamp=bundle.summary.run_timestamp,
            output_dir=artifacts_dir,
        )
        manager.write_text(metrics_json_path, metrics_df.to_json(orient="records", indent=2))
        artifact_bundle.metrics_json_path = metrics_json_path

    if bundle.compare_leaderboard:
        compare_df = leaderboard_to_dataframe(bundle.compare_leaderboard)
        compare_csv_path = manager.build_artifact_path(
            kind=ArtifactKind.EXPERIMENT,
            stem=bundle.dataset_name,
            label="experiment_compare",
            suffix=".csv",
            timestamp=bundle.summary.run_timestamp,
            output_dir=artifacts_dir,
        )
        manager.write_dataframe_csv(compare_csv_path, compare_df, index=False)
        artifact_bundle.compare_csv_path = compare_csv_path

        compare_json_path = manager.build_artifact_path(
            kind=ArtifactKind.EXPERIMENT,
            stem=bundle.dataset_name,
            label="experiment_compare",
            suffix=".json",
            timestamp=bundle.summary.run_timestamp,
            output_dir=artifacts_dir,
        )
        manager.write_json(compare_json_path, [row.model_dump(mode="json") for row in bundle.compare_leaderboard])
        artifact_bundle.compare_json_path = compare_json_path

    if bundle.tuned_result is not None:
        tune_json_path = manager.build_artifact_path(
            kind=ArtifactKind.EXPERIMENT,
            stem=bundle.dataset_name,
            label="experiment_tune",
            suffix=".json",
            timestamp=bundle.summary.run_timestamp,
            output_dir=artifacts_dir,
        )
        manager.write_text(tune_json_path, bundle.tuned_result.model_dump_json(indent=2))
        artifact_bundle.tune_json_path = tune_json_path

    summary_json_path = manager.build_artifact_path(
        kind=ArtifactKind.EXPERIMENT,
        stem=bundle.dataset_name,
        label="experiment_summary",
        suffix=".json",
        timestamp=bundle.summary.run_timestamp,
        output_dir=artifacts_dir,
    )
    manager.write_text(summary_json_path, bundle.summary.model_dump_json(indent=2))
    artifact_bundle.summary_json_path = summary_json_path

    markdown_summary_path = manager.build_artifact_path(
        kind=ArtifactKind.EXPERIMENT,
        stem=bundle.dataset_name,
        label="experiment_summary",
        suffix=".md",
        timestamp=bundle.summary.run_timestamp,
        output_dir=artifacts_dir,
    )
    manager.write_text(markdown_summary_path, _render_markdown(bundle))
    artifact_bundle.markdown_summary_path = markdown_summary_path

    if bundle.saved_model_metadata is not None:
        model_metadata_path = manager.build_artifact_path(
            kind=ArtifactKind.EXPERIMENT,
            stem=bundle.dataset_name,
            label="saved_model_metadata",
            suffix=".json",
            timestamp=bundle.summary.run_timestamp,
            output_dir=artifacts_dir,
        )
        manager.write_text(model_metadata_path, bundle.saved_model_metadata.model_dump_json(indent=2))
        artifact_bundle.saved_model_metadata_path = model_metadata_path
        if model_metadata_path not in artifact_bundle.saved_model_metadata_paths:
            artifact_bundle.saved_model_metadata_paths.append(model_metadata_path)

        if bundle.saved_model_metadata.experiment_snapshot_path is not None:
            snapshot_metadata_path = manager.build_artifact_path(
                kind=ArtifactKind.EXPERIMENT,
                stem=bundle.dataset_name,
                label="experiment_snapshot_metadata",
                suffix=".json",
                timestamp=bundle.summary.run_timestamp,
                output_dir=artifacts_dir,
            )
            manager.write_json(
                snapshot_metadata_path,
                {
                    "snapshot_path": str(bundle.saved_model_metadata.experiment_snapshot_path),
                    "includes_original_data": False,
                },
            )
            artifact_bundle.experiment_snapshot_metadata_path = snapshot_metadata_path

    return artifact_bundle


def _render_markdown(bundle: ExperimentResultBundle) -> str:
    summary = bundle.summary
    lines = [f"# Experiment Summary - {summary.dataset_name or 'Dataset'}", ""]
    lines.append(f"- Task type: {summary.task_type.value}")
    lines.append(f"- Target column: {summary.target_column}")
    if summary.compare_optimize_metric:
        lines.append(
            f"- Compare metric: {summary.compare_optimize_metric} ({summary.compare_ranking_direction.value})"
        )
    if summary.best_baseline_model_name:
        lines.append(f"- Best baseline: {summary.best_baseline_model_name} ({summary.best_baseline_score})")
    if summary.tuned_model_name:
        lines.append(f"- Tuned model: {summary.tuned_model_name} ({summary.tuned_score})")
    lines.append(f"- Duration: {summary.experiment_duration_seconds:.2f}s")
    lines.append("")

    if bundle.compare_leaderboard:
        lines.append("## Leaderboard")
        lines.append("")
        lines.append("| Rank | Model | Score |")
        lines.append("|------|-------|-------|")
        for row in bundle.compare_leaderboard[:10]:
            lines.append(f"| {row.rank} | {row.model_name} | {row.primary_score} |")
        lines.append("")

    if summary.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in summary.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)