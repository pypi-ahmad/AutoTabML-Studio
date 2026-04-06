"""Summary helpers for the tracking layer."""

from __future__ import annotations

from app.tracking.schemas import RunHistoryItem


def run_summary_line(run: RunHistoryItem) -> str:
    """Return a compact one-line summary of a run."""

    parts = [f"[{run.run_type.value}]"]
    if run.run_name:
        parts.append(run.run_name)
    if run.task_type:
        parts.append(f"task={run.task_type}")
    if run.model_name:
        parts.append(f"model={run.model_name}")
    if run.primary_metric_name and run.primary_metric_value is not None:
        parts.append(f"{run.primary_metric_name}={run.primary_metric_value}")
    if run.duration_seconds is not None:
        parts.append(f"{run.duration_seconds:.1f}s")
    return " | ".join(parts)
