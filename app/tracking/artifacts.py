"""Artifact generation for comparison bundles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.artifacts import ArtifactKind, LocalArtifactManager
from app.tracking.schemas import ComparisonBundle


def write_comparison_artifacts(
    bundle: ComparisonBundle,
    output_dir: Path,
) -> dict[str, Path]:
    """Write comparison artifacts and return their paths."""

    manager = LocalArtifactManager()
    left_id = bundle.left.run_id[:8]
    right_id = bundle.right.run_id[:8]
    stem = f"compare_{left_id}_vs_{right_id}"
    paths: dict[str, Path] = {}

    comparison_json_path = manager.build_artifact_path(
        kind=ArtifactKind.COMPARISON,
        stem=stem,
        suffix=".json",
        output_dir=output_dir,
    )
    manager.write_text(comparison_json_path, bundle.model_dump_json(indent=2))
    paths["comparison_json"] = comparison_json_path

    if bundle.metric_deltas:
        rows = []
        for delta in bundle.metric_deltas:
            rows.append({
                "Metric": delta.name,
                "Left": delta.left_value,
                "Right": delta.right_value,
                "Delta": delta.delta,
                "Better": delta.better_side or "",
            })
        metrics_csv_path = manager.build_artifact_path(
            kind=ArtifactKind.COMPARISON,
            stem=stem,
            label="metrics",
            suffix=".csv",
            output_dir=output_dir,
        )
        manager.write_dataframe_csv(metrics_csv_path, pd.DataFrame(rows), index=False)
        paths["metrics_csv"] = metrics_csv_path

    markdown_path = manager.build_artifact_path(
        kind=ArtifactKind.COMPARISON,
        stem=stem,
        suffix=".md",
        output_dir=output_dir,
    )
    manager.write_text(markdown_path, _render_markdown(bundle))
    paths["markdown"] = markdown_path

    return paths


def _render_markdown(bundle: ComparisonBundle) -> str:
    lines = [
        "# Run Comparison",
        "",
        f"- Left run: `{bundle.left.run_id}` ({bundle.left.run_name or 'unnamed'})",
        f"- Right run: `{bundle.right.run_id}` ({bundle.right.run_name or 'unnamed'})",
        f"- Comparable: {'Yes' if bundle.comparable else 'No'}",
        "",
    ]

    if bundle.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in bundle.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    if bundle.metric_deltas:
        lines.append("## Metrics")
        lines.append("")
        lines.append("| Metric | Left | Right | Delta | Better |")
        lines.append("|--------|------|-------|-------|--------|")
        for delta in bundle.metric_deltas:
            left_str = f"{delta.left_value:.4f}" if delta.left_value is not None else "N/A"
            right_str = f"{delta.right_value:.4f}" if delta.right_value is not None else "N/A"
            delta_str = f"{delta.delta:+.4f}" if delta.delta is not None else ""
            lines.append(f"| {delta.name} | {left_str} | {right_str} | {delta_str} | {delta.better_side or ''} |")
        lines.append("")

    if bundle.config_differences:
        lines.append("## Configuration Differences")
        lines.append("")
        lines.append("| Key | Left | Right |")
        lines.append("|-----|------|-------|")
        for diff in bundle.config_differences:
            lines.append(f"| {diff.key} | {diff.left_value or 'N/A'} | {diff.right_value or 'N/A'} |")
        lines.append("")

    return "\n".join(lines)
