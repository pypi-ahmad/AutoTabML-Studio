"""Artifact generation for prediction jobs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.artifacts import ArtifactKind, LocalArtifactManager
from app.prediction.errors import PredictionArtifactError
from app.prediction.schemas import LoadedModel, PredictionArtifactBundle, PredictionSummary


def write_prediction_artifacts(
    *,
    loaded_model: LoadedModel,
    scored_dataframe: pd.DataFrame,
    summary: PredictionSummary,
    output_dir: Path,
    output_path: Path | None = None,
    output_stem: str | None = None,
) -> PredictionArtifactBundle:
    """Persist prediction artifacts and return their paths."""

    try:
        manager = LocalArtifactManager()
        stem = output_stem or summary.input_source or summary.model_identifier

        scored_csv_path = output_path or manager.build_artifact_path(
            kind=ArtifactKind.PREDICTION,
            stem=stem,
            label=summary.mode.value,
            suffix=".csv",
            timestamp=summary.run_timestamp,
            output_dir=output_dir,
        )
        summary.output_artifact_path = scored_csv_path
        manager.write_dataframe_csv(scored_csv_path, scored_dataframe, index=True)

        summary_json_path = manager.build_artifact_path(
            kind=ArtifactKind.PREDICTION,
            stem=stem,
            label=f"{summary.mode.value}_summary",
            suffix=".json",
            timestamp=summary.run_timestamp,
            output_dir=output_dir,
        )
        manager.write_text(summary_json_path, summary.model_dump_json(indent=2))

        metadata_json_path = manager.build_artifact_path(
            kind=ArtifactKind.PREDICTION,
            stem=stem,
            label=f"{summary.mode.value}_metadata",
            suffix=".json",
            timestamp=summary.run_timestamp,
            output_dir=output_dir,
        )
        manager.write_json(
            metadata_json_path,
            {
                "model_source": loaded_model.source_type.value,
                "model_identifier": loaded_model.model_identifier,
                "load_reference": loaded_model.load_reference,
                "task_type": loaded_model.task_type.value,
                "loader_name": loaded_model.loader_name,
                "scorer_kind": loaded_model.scorer_kind,
                "feature_columns": loaded_model.feature_columns,
                "target_column": loaded_model.target_column,
                "supported_prediction_modes": [mode.value for mode in loaded_model.supported_prediction_modes],
                "metadata": loaded_model.metadata,
                "input_shape": list(scored_dataframe.shape),
                "timestamp": summary.run_timestamp.isoformat(),
            },
        )

        markdown_path = manager.build_artifact_path(
            kind=ArtifactKind.PREDICTION,
            stem=stem,
            label=summary.mode.value,
            suffix=".md",
            timestamp=summary.run_timestamp,
            output_dir=output_dir,
        )
        manager.write_text(markdown_path, _render_markdown(summary, loaded_model))

        return PredictionArtifactBundle(
            scored_csv_path=scored_csv_path,
            summary_json_path=summary_json_path,
            metadata_json_path=metadata_json_path,
            markdown_summary_path=markdown_path,
        )
    except Exception as exc:
        raise PredictionArtifactError(f"Could not write prediction artifacts: {exc}") from exc


def _render_markdown(summary: PredictionSummary, loaded_model: LoadedModel) -> str:
    lines = ["# Prediction Summary", ""]
    lines.append(f"- Status: {summary.status.value}")
    lines.append(f"- Mode: {summary.mode.value}")
    lines.append(f"- Model source: {loaded_model.source_type.value}")
    lines.append(f"- Model identifier: {loaded_model.model_identifier}")
    lines.append(f"- Load reference: `{loaded_model.load_reference}`")
    lines.append(f"- Task type: {loaded_model.task_type.value}")
    lines.append(f"- Input source: {summary.input_source}")
    lines.append(f"- Rows scored: {summary.rows_scored}")
    if summary.output_artifact_path is not None:
        lines.append(f"- Scored CSV: `{summary.output_artifact_path}`")
    if summary.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for warning in summary.warnings:
            lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)