"""Produce validation artifacts: JSON summary, Markdown report."""

from __future__ import annotations

import logging
from pathlib import Path

from app.artifacts import ArtifactKind, LocalArtifactManager
from app.validation.schemas import ValidationArtifactBundle, ValidationResultSummary

logger = logging.getLogger(__name__)


def write_artifacts(
    summary: ValidationResultSummary,
    artifacts_dir: Path,
) -> ValidationArtifactBundle:
    """Write validation artifacts to disk and return the bundle."""
    manager = LocalArtifactManager()
    bundle = ValidationArtifactBundle()

    json_path = manager.build_artifact_path(
        kind=ArtifactKind.VALIDATION,
        stem=summary.dataset_name,
        label="validation",
        suffix=".json",
        timestamp=summary.run_timestamp,
        output_dir=artifacts_dir,
    )
    manager.write_text(json_path, summary.model_dump_json(indent=2))
    bundle.summary_json_path = json_path
    logger.info("Validation JSON written to %s", json_path)

    md_path = manager.build_artifact_path(
        kind=ArtifactKind.VALIDATION,
        stem=summary.dataset_name,
        label="validation",
        suffix=".md",
        timestamp=summary.run_timestamp,
        output_dir=artifacts_dir,
    )
    manager.write_text(md_path, _render_markdown(summary))
    bundle.markdown_report_path = md_path
    logger.info("Validation Markdown written to %s", md_path)

    return bundle


def _render_markdown(summary: ValidationResultSummary) -> str:
    lines: list[str] = []
    lines.append(f"# Validation Report – {summary.dataset_name or 'Dataset'}")
    lines.append("")
    lines.append(f"**Run:** {summary.run_timestamp.isoformat()}")
    lines.append(f"**Rows:** {summary.row_count} | **Columns:** {summary.column_count}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total checks | {summary.total_checks} |")
    lines.append(f"| Passed | {summary.passed_count} |")
    lines.append(f"| Warnings | {summary.warning_count} |")
    lines.append(f"| Failed | {summary.failed_count} |")
    lines.append("")

    if summary.checks:
        lines.append("## Check Details")
        lines.append("")
        lines.append("| Check | Severity | Status | Message |")
        lines.append("|-------|----------|--------|---------|")
        for c in summary.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"| {c.check_name} | {c.severity.value} | {status} | {c.message} |")
        lines.append("")

    return "\n".join(lines)
