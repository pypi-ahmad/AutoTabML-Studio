"""Streamlit page for job and dataset run history."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from app.pages.dataset_workspace import go_to_page
from app.pages.ui_labels import format_enum_value, render_metadata_table
from app.security.masking import safe_error_message
from app.state.session import get_or_init_state
from app.storage import build_metadata_store
from app.storage.models import AppJobType
from app.tracking.description_generator import generate_llm_description, generate_template_description

logger = logging.getLogger(__name__)


def render_history_page() -> None:
    state = get_or_init_state()
    metadata_store = build_metadata_store(state.settings)
    st.title("📋 History")
    st.caption(
        "Every time you run a workflow (validation, profiling, benchmark, experiment, or prediction), "
        "it's recorded here so you can review and compare past results."
    )

    if metadata_store is None:
        st.info(
            "Run history is not available yet. "
            "Please complete the initial setup to enable job history (see the README or ask your administrator)."
        )
        return

    # ── Filters ────────────────────────────────────────────────────────
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        job_type_value = st.selectbox(
            "Workflow type",
            options=["All", "Validation", "Profiling", "Benchmark", "Experiment", "FLAML", "Prediction"],
            index=0,
            key="hist_job_type",
            help="Filter by the type of workflow that was run.",
        )
    with filter_col2:
        dataset_filter = st.text_input("Dataset name (contains)", value="", key="hist_ds_filter", help="Type part of a dataset name to narrow down results.").strip()
    with filter_col3:
        limit = int(
            st.number_input("Max results", min_value=5, max_value=500, value=50, step=10, key="hist_limit", help="How many past runs to show at most.")
        )

    # ── Fetch jobs ─────────────────────────────────────────────────────
    job_type_filter = None
    if job_type_value != "All":
        job_type_filter = AppJobType(job_type_value.lower())

    try:
        jobs = metadata_store.list_recent_jobs(limit=limit, job_type=job_type_filter)
    except Exception as exc:
        st.error(f"Failed to load history: {safe_error_message(exc)}")
        return

    # Client-side dataset name filter
    if dataset_filter:
        target = dataset_filter.lower()
        jobs = [j for j in jobs if j.dataset_name and target in j.dataset_name.lower()]

    if not jobs:
        if job_type_filter or dataset_filter:
            st.info("No jobs match the current filters. Try clearing the filters above.")
        else:
            st.info(
                "**No activity yet.**\n\n"
                "Every time you run a workflow — validation, profiling, benchmark, experiment, or prediction — "
                "it gets recorded here.\n\n"
                "**Start by loading a dataset**, then run your first benchmark."
            )
            if st.button("📥 Load Data", key="hist_goto_load", type="primary"):
                go_to_page("Load Data")
        return

    st.caption(f"Showing **{len(jobs)}** job(s)")

    # ── Executive summary ──────────────────────────────────────────────
    _success = sum(1 for j in jobs if j.status.value.lower() == "success")
    _failed = sum(1 for j in jobs if j.status.value.lower() == "failed")
    _datasets_touched = {j.dataset_name for j in jobs if j.dataset_name}
    _job_types = {format_enum_value(j.job_type.value) for j in jobs}
    st.info(
        f"**Summary:** **{_success}** successful, **{_failed}** failed "
        f"across **{len(_datasets_touched)}** dataset(s). "
        f"Workflow types: {', '.join(sorted(_job_types))}."
    )

    # ── Main table ─────────────────────────────────────────────────────
    rows = []
    for job in jobs:
        rows.append({
            "Dataset": job.dataset_name or "—",
            "Job": format_enum_value(job.job_type.value),
            "Status": _status_badge(job.status.value),
            "Title": job.title or "",
            "Updated": job.updated_at,
            "Created": job.created_at,
        })
    _history_df = pd.DataFrame(rows)
    st.dataframe(_history_df, width="stretch", hide_index=True)

    # ── Export ─────────────────────────────────────────────────────────
    st.download_button(
        "📤 Download History CSV",
        data=_history_df.to_csv(index=False).encode("utf-8"),
        file_name="autotabml_history.csv",
        mime="text/csv",
        key="hist_dl_csv",
    )

    # ── Job detail ─────────────────────────────────────────────────────
    job_labels = [
        f"{j.dataset_name or '—'} · {format_enum_value(j.job_type.value)} · {_status_icon(j.status.value)} ({j.updated_at:%Y-%m-%d %H:%M})"
        for j in jobs
    ]
    selected_label = st.selectbox("Inspect job", options=job_labels, key="hist_inspect", help="Pick a past job to see its details below.")
    if selected_label:
        selected_job = jobs[job_labels.index(selected_label)]

        st.subheader("Job Detail")

        # Decision-support: one-line purpose
        _job_type_label = format_enum_value(selected_job.job_type.value)
        _status_label = format_enum_value(selected_job.status.value)
        _ds_name = selected_job.dataset_name or 'N/A'
        _purpose = f"{_job_type_label} on **{_ds_name}** — {_status_label.lower()}"
        st.caption(_purpose)

        d1, d2, d3 = st.columns(3)
        d1.metric("Type", format_enum_value(selected_job.job_type.value))
        d2.metric("Status", format_enum_value(selected_job.status.value))
        d3.metric("Updated", f"{selected_job.updated_at:%Y-%m-%d %H:%M}")

        if selected_job.title:
            st.caption(f"Title: {selected_job.title}")

        _has_tech_details = (
            selected_job.primary_artifact_path
            or selected_job.summary_path
            or selected_job.mlflow_run_id
        )
        if _has_tech_details:
            with st.expander("Technical details", expanded=False):
                if selected_job.primary_artifact_path:
                    artifact_path = Path(selected_job.primary_artifact_path)
                    st.caption(f"Primary output file: `{artifact_path.name}`")
                if selected_job.summary_path:
                    st.caption(f"Summary file: `{Path(selected_job.summary_path).name}`")
                if selected_job.mlflow_run_id:
                    st.caption(f"Tracking run ID: `{selected_job.mlflow_run_id}` — use this to look up the run in MLflow.")

        if selected_job.metadata:
            with st.expander("Run details", expanded=False):
                _meta = selected_job.metadata if isinstance(selected_job.metadata, dict) else {"value": selected_job.metadata}
                render_metadata_table(_meta)

        # ── MLflow Run Description ─────────────────────────────────────
        _render_job_description(selected_job, state)

    # ── MLflow runs (optional advanced section) ────────────────────────
    _render_mlflow_section(state.settings.tracking)


def _render_job_description(job, state) -> None:  # noqa: ANN001
    """Render the MLflow run description panel for a job."""
    settings = state.settings

    if not settings.mlflow_descriptions_enabled:
        return

    meta = job.metadata or {}

    # Check if a cached LLM description exists in metadata
    cached_desc = meta.get("mlflow_description")

    if cached_desc:
        with st.expander("📝 Run Summary", expanded=True):
            st.markdown(cached_desc)
        return

    # Generate description on demand
    with st.expander("📝 Run Summary", expanded=True):
        use_llm = settings.llm_descriptions_enabled
        gen_key = f"hist_gen_desc_{job.job_id}"

        if use_llm:
            api_key = state.get_provider_api_key(state.provider)
            model_id = state.selected_model_id
            if not api_key:
                from app.pages.ui_labels import PROVIDER_LABELS
                _provider_name = PROVIDER_LABELS.get(state.provider.value, state.provider.value.title())
                st.warning(
                    f"AI-powered summaries need a **{_provider_name}** API key. "
                    "You can set one in **Settings › Advanced**, or turn off AI summaries."
                )
                use_llm = False

        if use_llm:
            if st.button("🤖 Generate AI Summary", key=gen_key):
                with st.spinner("Generating summary…"):
                    try:
                        from app.providers.catalog_service import build_provider

                        provider = build_provider(
                            state.provider,
                            api_key=api_key,
                            base_url=(
                                settings.ollama_base_url
                                if state.provider.value == "ollama" else None
                            ),
                        )
                        desc = generate_llm_description(
                            job.job_type,
                            dataset_name=job.dataset_name,
                            metadata=meta,
                            mlflow_run_id=job.mlflow_run_id,
                            provider=provider,
                            model_id=model_id,
                        )
                    except Exception as exc:
                        logger.warning("LLM generation failed: %s", exc)
                        desc = generate_template_description(
                            job.job_type,
                            dataset_name=job.dataset_name,
                            metadata=meta,
                            mlflow_run_id=job.mlflow_run_id,
                        )
                st.markdown(desc)
                _save_description_to_job(job, desc, state)
            else:
                # Show template as preview
                desc = generate_template_description(
                    job.job_type,
                    dataset_name=job.dataset_name,
                    metadata=meta,
                    mlflow_run_id=job.mlflow_run_id,
                )
                st.markdown(desc)
                st.caption("💡 Click the button above to generate an AI-powered summary.")
        else:
            # Template-only mode
            desc = generate_template_description(
                job.job_type,
                dataset_name=job.dataset_name,
                metadata=meta,
                mlflow_run_id=job.mlflow_run_id,
            )
            st.markdown(desc)


def _save_description_to_job(job, description: str, state) -> None:  # noqa: ANN001
    """Persist the generated description into job metadata."""
    try:
        metadata_store = build_metadata_store(state.settings)
        if metadata_store is None:
            return
        meta = dict(job.metadata or {})
        meta["mlflow_description"] = description
        job.metadata = meta
        metadata_store.record_job(job)
    except Exception:
        logger.debug("Could not save description to job metadata", exc_info=True)


def _render_mlflow_section(tracking_settings) -> None:  # noqa: ANN001
    """Optional collapsible section to browse raw MLflow runs."""

    from app.tracking.mlflow_query import is_mlflow_available

    if not is_mlflow_available():
        return

    with st.expander("Experiment Tracking Runs (Advanced)", expanded=False):
        from app.tracking.history_service import HistoryService

        service = HistoryService(
            tracking_uri=tracking_settings.tracking_uri,
            default_experiment_names=tracking_settings.default_experiment_names,
            default_limit=tracking_settings.history_page_default_limit,
        )

        mlflow_limit = int(
            st.number_input("MLflow max results", min_value=1, max_value=200, value=20, step=10, key="hist_mlflow_limit", help="How many MLflow runs to fetch.")
        )

        try:
            runs = service.list_runs(limit=mlflow_limit)
        except Exception as exc:
            st.error(f"Failed to query MLflow: {safe_error_message(exc)}")
            return

        if not runs:
            st.caption("No MLflow runs found.")
            return

        ml_rows = []
        for run in runs:
            ml_rows.append({
                "Dataset": run.dataset_name or "—",
                "Type": format_enum_value(run.run_type.value),
                "Status": format_enum_value(run.status.value),
                "Model": run.model_name or "",
                "Score": f"{run.primary_metric_value:.4f}" if run.primary_metric_value is not None else "",
                "Metric": run.primary_metric_name or "",
                "Duration": f"{run.duration_seconds:.1f}s" if run.duration_seconds else "",
            })
        st.dataframe(pd.DataFrame(ml_rows), width="stretch", hide_index=True)


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
