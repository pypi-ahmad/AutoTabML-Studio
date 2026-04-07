"""Shared UI helpers for rendering job history and model details on workflow pages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.security.masking import safe_error_message
from app.pages.ui_labels import render_metadata_table, format_enum_value, PREDICTION_TASK_TYPE_LABELS
from app.storage.models import AppJobType


def render_past_runs_section(
    metadata_store,  # noqa: ANN001
    job_type: AppJobType,
    *,
    key_prefix: str,
    limit: int = 10,
) -> None:
    """Render a 'Past Runs' expander showing job history for a specific job type."""

    if metadata_store is None:
        return

    try:
        jobs = metadata_store.list_recent_jobs(limit=limit, job_type=job_type)
    except Exception:
        return

    if not jobs:
        return

    label = format_enum_value(job_type.value)
    with st.expander(f"📋 Past {label} Runs ({len(jobs)})", expanded=False):
        rows = []
        for job in jobs:
            rows.append({
                "Dataset": job.dataset_name or "—",
                "Status": _status_badge(job.status.value),
                "Title": job.title or "",
                "Updated": job.updated_at,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # Detail selector
        job_labels = [
            f"{j.dataset_name or '—'} · {_status_icon(j.status.value)} ({j.updated_at:%Y-%m-%d %H:%M})"
            for j in jobs
        ]
        selected = st.selectbox(
            "Inspect run",
            options=job_labels,
            key=f"{key_prefix}_past_run_select",
            help="Pick a past run to see its details below.",
        )
        if selected:
            job = jobs[job_labels.index(selected)]
            d1, d2, d3 = st.columns(3)
            d1.metric("Dataset", job.dataset_name or "N/A")
            d2.metric("Status", format_enum_value(job.status.value))
            d3.metric("Updated", f"{job.updated_at:%Y-%m-%d %H:%M}")

            if job.primary_artifact_path or job.summary_path or job.mlflow_run_id:
                with st.expander("Technical details", expanded=False):
                    if job.primary_artifact_path:
                        st.caption(f"Output file: `{Path(job.primary_artifact_path).name}`")
                    if job.summary_path:
                        st.caption(f"Summary file: `{Path(job.summary_path).name}`")
                    if job.mlflow_run_id:
                        st.caption(f"Tracking run ID: `{job.mlflow_run_id}` — use this to look up the run in MLflow.")
            if job.metadata:
                with st.expander("Run details", expanded=False):
                    _meta = job.metadata if isinstance(job.metadata, dict) else {"value": job.metadata}
                    render_metadata_table(_meta)


def render_saved_models_section(
    prediction_settings,  # noqa: ANN001
    *,
    key_prefix: str,
) -> None:
    """Render a 'Saved Models' expander with model discovery and detail cards."""

    from app.prediction.selectors import discover_local_saved_models, PredictionTaskType

    refs = discover_local_saved_models(
        model_dirs=prediction_settings.supported_local_model_dirs,
        metadata_dirs=prediction_settings.local_model_metadata_dirs,
    )

    if not refs:
        return

    with st.expander(f"🗂️ Saved Models ({len(refs)})", expanded=False):
        labels = [
            f"{r.display_name} ({PREDICTION_TASK_TYPE_LABELS.get(r.task_type.value, format_enum_value(r.task_type.value))})"
            for r in refs
        ]
        selected = st.selectbox(
            "Select model",
            options=labels,
            key=f"{key_prefix}_saved_model_select",
            help="Pick a saved model to see its details.",
        )
        if selected:
            ref = refs[labels.index(selected)]
            c1, c2, c3 = st.columns(3)
            task_label = PREDICTION_TASK_TYPE_LABELS.get(ref.task_type.value, format_enum_value(ref.task_type.value)) if ref.task_type != PredictionTaskType.UNKNOWN else "Unknown"
            c1.metric("Task", task_label)
            c2.metric("Features", len(ref.feature_columns))
            target = ref.metadata.get("target_column", "—")
            c3.metric("Target", target)

            with st.expander("Saved files", expanded=False):
                st.caption(f"Model file: **{ref.model_path.name}**")
            if ref.metadata.get("dataset_fingerprint"):
                with st.expander("Training data details", expanded=False):
                    st.caption(f"Dataset version: `{ref.metadata['dataset_fingerprint'][:16]}…` — a unique ID for the exact data this model was trained on.")
            if ref.feature_columns:
                with st.expander("Input columns", expanded=False):
                    st.code(", ".join(ref.feature_columns), language="text")


def _status_badge(status: str) -> str:
    if status.lower() == "success":
        return "✅ Success"
    if status.lower() == "failed":
        return "❌ Failed"
    return status.title()


def _status_icon(status: str) -> str:
    if status.lower() == "success":
        return "✅"
    if status.lower() == "failed":
        return "❌"
    return "⏳"
