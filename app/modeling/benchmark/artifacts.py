"""Artifact generation for benchmark runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.artifacts import ArtifactKind, LocalArtifactManager
from app.modeling.benchmark.schemas import BenchmarkArtifactBundle, BenchmarkResultBundle
from app.modeling.benchmark.summary import leaderboard_to_dataframe

logger = logging.getLogger(__name__)


def write_benchmark_artifacts(
    bundle: BenchmarkResultBundle,
    artifacts_dir: Path,
) -> BenchmarkArtifactBundle:
    """Write benchmark artifacts to disk and return their paths."""

    manager = LocalArtifactManager()
    artifact_bundle = BenchmarkArtifactBundle()

    raw_csv_path = manager.build_artifact_path(
        kind=ArtifactKind.BENCHMARK,
        stem=bundle.dataset_name,
        label="benchmark_raw",
        suffix=".csv",
        timestamp=bundle.summary.run_timestamp,
        output_dir=artifacts_dir,
    )
    manager.write_dataframe_csv(raw_csv_path, bundle.raw_results, index=True)
    artifact_bundle.raw_results_csv_path = raw_csv_path

    leaderboard_df = leaderboard_to_dataframe(bundle.leaderboard)

    leaderboard_csv_path = manager.build_artifact_path(
        kind=ArtifactKind.BENCHMARK,
        stem=bundle.dataset_name,
        label="benchmark_leaderboard",
        suffix=".csv",
        timestamp=bundle.summary.run_timestamp,
        output_dir=artifacts_dir,
    )
    manager.write_dataframe_csv(leaderboard_csv_path, leaderboard_df, index=False)
    artifact_bundle.leaderboard_csv_path = leaderboard_csv_path

    leaderboard_json_path = manager.build_artifact_path(
        kind=ArtifactKind.BENCHMARK,
        stem=bundle.dataset_name,
        label="benchmark_leaderboard",
        suffix=".json",
        timestamp=bundle.summary.run_timestamp,
        output_dir=artifacts_dir,
    )
    manager.write_json(leaderboard_json_path, [row.model_dump(mode="json") for row in bundle.leaderboard])
    artifact_bundle.leaderboard_json_path = leaderboard_json_path

    summary_json_path = manager.build_artifact_path(
        kind=ArtifactKind.BENCHMARK,
        stem=bundle.dataset_name,
        label="benchmark_summary",
        suffix=".json",
        timestamp=bundle.summary.run_timestamp,
        output_dir=artifacts_dir,
    )
    manager.write_text(summary_json_path, bundle.summary.model_dump_json(indent=2))
    artifact_bundle.summary_json_path = summary_json_path

    markdown_path = manager.build_artifact_path(
        kind=ArtifactKind.BENCHMARK,
        stem=bundle.dataset_name,
        label="benchmark_summary",
        suffix=".md",
        timestamp=bundle.summary.run_timestamp,
        output_dir=artifacts_dir,
    )
    manager.write_text(markdown_path, _render_markdown(bundle))
    artifact_bundle.markdown_summary_path = markdown_path

    score_chart_path = _write_score_chart(bundle, artifacts_dir, bundle.dataset_name, bundle.summary.run_timestamp)
    if score_chart_path is not None:
        artifact_bundle.score_chart_path = score_chart_path

    time_chart_path = _write_time_chart(bundle, artifacts_dir, bundle.dataset_name, bundle.summary.run_timestamp)
    if time_chart_path is not None:
        artifact_bundle.training_time_chart_path = time_chart_path

    return artifact_bundle


def _render_markdown(bundle: BenchmarkResultBundle) -> str:
    summary = bundle.summary
    lines: list[str] = []
    lines.append(f"# Benchmark Summary - {summary.dataset_name or 'Dataset'}")
    lines.append("")
    lines.append(f"- Task type: {summary.task_type.value}")
    lines.append(f"- Target column: {summary.target_column}")
    lines.append(f"- Ranking metric: {summary.ranking_metric} ({summary.ranking_direction.value})")
    lines.append(f"- Models evaluated: {summary.model_count}")
    lines.append(f"- Benchmark duration: {summary.benchmark_duration_seconds:.2f}s")
    lines.append(f"- Train/Test rows: {summary.train_row_count}/{summary.test_row_count}")
    if summary.best_model_name is not None:
        lines.append(f"- Best model: {summary.best_model_name} ({summary.best_score})")
    if summary.fastest_model_name is not None:
        lines.append(
            f"- Fastest model: {summary.fastest_model_name} ({summary.fastest_model_time_seconds:.4f}s)"
        )
    if summary.sampled_row_count is not None:
        lines.append(f"- Sampled rows used for benchmark: {summary.sampled_row_count}")
    lines.append("")

    if summary.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in summary.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    if bundle.top_models:
        lines.append("## Top Models")
        lines.append("")
        lines.append("| Rank | Model | Primary Score | Time (s) |")
        lines.append("|------|-------|---------------|----------|")
        for row in bundle.top_models:
            lines.append(
                f"| {row.rank} | {row.model_name} | {row.primary_score} | {row.training_time_seconds} |"
            )
        lines.append("")

    return "\n".join(lines)


def _write_score_chart(
    bundle: BenchmarkResultBundle,
    artifacts_dir: Path,
    dataset_name: str | None,
    timestamp,
) -> Path | None:
    if not bundle.top_models:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    manager = LocalArtifactManager()
    chart_path = manager.build_artifact_path(
        kind=ArtifactKind.BENCHMARK,
        stem=dataset_name,
        label="benchmark_scores",
        suffix=".png",
        timestamp=timestamp,
        output_dir=artifacts_dir,
    )
    top_models = bundle.top_models[:10]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([row.model_name for row in reversed(top_models)], [row.primary_score or 0.0 for row in reversed(top_models)])
    ax.set_title(f"Top Benchmark Models by {bundle.summary.ranking_metric}")
    ax.set_xlabel(bundle.summary.ranking_metric)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def _write_time_chart(
    bundle: BenchmarkResultBundle,
    artifacts_dir: Path,
    dataset_name: str | None,
    timestamp,
) -> Path | None:
    timed_models = [row for row in bundle.top_models if row.training_time_seconds is not None]
    if not timed_models:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    manager = LocalArtifactManager()
    chart_path = manager.build_artifact_path(
        kind=ArtifactKind.BENCHMARK,
        stem=dataset_name,
        label="benchmark_times",
        suffix=".png",
        timestamp=timestamp,
        output_dir=artifacts_dir,
    )
    top_models = timed_models[:10]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([row.model_name for row in reversed(top_models)], [row.training_time_seconds or 0.0 for row in reversed(top_models)])
    ax.set_title("Top Benchmark Model Training Times")
    ax.set_xlabel("Time Taken (seconds)")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path