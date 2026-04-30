"""Artifact generation for experiment runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.artifacts import ArtifactKind
from app.modeling.base import BaseArtifacts
from app.modeling.pycaret.schemas import ExperimentArtifactBundle, ExperimentResultBundle
from app.modeling.pycaret.summary import leaderboard_to_dataframe
from app.security.trusted_artifacts import write_checksum_file


class ExperimentArtifactsWriter(BaseArtifacts[ExperimentResultBundle, ExperimentArtifactBundle]):
    """Shared-path artifact writer for experiment result bundles."""

    artifact_kind = ArtifactKind.EXPERIMENT
    artifact_bundle_cls = ExperimentArtifactBundle

    def build(self) -> ExperimentArtifactBundle:
        self.artifacts.plot_artifacts = list(self.bundle.evaluation_plots)
        if self.bundle.saved_model_artifacts:
            self.artifacts.saved_model_metadata_paths = [
                artifact.metadata_path
                for artifact in self.bundle.saved_model_artifacts
                if artifact.metadata_path is not None
            ]

        setup_json_path = self._artifact_path(label="experiment_setup", suffix=".json")
        self._write_text(setup_json_path, self.bundle.summary.setup_config.model_dump_json(indent=2))
        self.artifacts.setup_json_path = setup_json_path

        if self.bundle.available_metrics:
            metrics_df = pd.DataFrame([row.model_dump(mode="json") for row in self.bundle.available_metrics])
            metrics_csv_path = self._artifact_path(label="experiment_metrics", suffix=".csv")
            self._write_dataframe(metrics_csv_path, metrics_df, index=False)
            self.artifacts.metrics_csv_path = metrics_csv_path

            metrics_json_path = self._artifact_path(label="experiment_metrics", suffix=".json")
            self._write_text(metrics_json_path, metrics_df.to_json(orient="records", indent=2))
            self.artifacts.metrics_json_path = metrics_json_path

        if self.bundle.compare_leaderboard:
            compare_df = leaderboard_to_dataframe(self.bundle.compare_leaderboard)
            compare_csv_path = self._artifact_path(label="experiment_compare", suffix=".csv")
            self._write_dataframe(compare_csv_path, compare_df, index=False)
            self.artifacts.compare_csv_path = compare_csv_path

            compare_json_path = self._artifact_path(label="experiment_compare", suffix=".json")
            self._write_json(
                compare_json_path,
                [row.model_dump(mode="json") for row in self.bundle.compare_leaderboard],
            )
            self.artifacts.compare_json_path = compare_json_path

        if self.bundle.tuned_result is not None:
            tune_json_path = self._artifact_path(label="experiment_tune", suffix=".json")
            self._write_text(tune_json_path, self.bundle.tuned_result.model_dump_json(indent=2))
            self.artifacts.tune_json_path = tune_json_path

        summary_json_path = self._artifact_path(label="experiment_summary", suffix=".json")
        self._write_text(summary_json_path, self.bundle.summary.model_dump_json(indent=2))
        self.artifacts.summary_json_path = summary_json_path

        markdown_summary_path = self._artifact_path(label="experiment_summary", suffix=".md")
        self._write_text(markdown_summary_path, _render_markdown(self.bundle))
        self.artifacts.markdown_summary_path = markdown_summary_path

        if self.bundle.saved_model_metadata is not None:
            model_metadata_path = self._artifact_path(label="saved_model_metadata", suffix=".json")
            self._write_text(model_metadata_path, self.bundle.saved_model_metadata.model_dump_json(indent=2))
            write_checksum_file(model_metadata_path)
            self.artifacts.saved_model_metadata_path = model_metadata_path
            if model_metadata_path not in self.artifacts.saved_model_metadata_paths:
                self.artifacts.saved_model_metadata_paths.append(model_metadata_path)

            if self.bundle.saved_model_metadata.experiment_snapshot_path is not None:
                snapshot_metadata_path = self._artifact_path(
                    label="experiment_snapshot_metadata",
                    suffix=".json",
                )
                self._write_json(
                    snapshot_metadata_path,
                    {
                        "snapshot_path": str(self.bundle.saved_model_metadata.experiment_snapshot_path),
                        "includes_original_data": False,
                    },
                )
                self.artifacts.experiment_snapshot_metadata_path = snapshot_metadata_path

        return self.artifacts


def write_experiment_artifacts(
    bundle: ExperimentResultBundle,
    artifacts_dir: Path,
) -> ExperimentArtifactBundle:
    """Write experiment artifacts to disk and return their paths."""

    return ExperimentArtifactsWriter(bundle, artifacts_dir).build()


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