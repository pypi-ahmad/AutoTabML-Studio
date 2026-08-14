"""Guided Auto Run page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.autorun import AutoRunConfig, AutoRunMode, plan_auto_run, suggest_targets
from app.background_jobs import BackgroundJobService
from app.pages.dataset_workspace import get_active_loaded_dataset, go_to_page
from app.pages.ui_cache import get_metadata_store
from app.state.session import get_or_init_state
from app.storage import AppJobStatus


def render_autorun_page() -> None:
    st.title("✨ Auto Run")
    st.caption("One guided path from a loaded table to an evaluated, explained, saved model.")
    state = get_or_init_state()
    store = get_metadata_store(state.settings)
    if store is None:
        st.error("Local metadata storage is unavailable.")
        return
    dataset_name, loaded = get_active_loaded_dataset(metadata_store=store)
    if loaded is None:
        st.info("Load a dataset before starting Auto Run.")
        if st.button("Load data", type="primary"):
            go_to_page("Load Data")
        return
    frame = loaded.dataframe
    st.success(f"Using **{dataset_name}** · {len(frame):,} rows · {len(frame.columns):,} columns")
    targets = suggest_targets(frame)
    with st.form("autorun_plan"):
        target = st.selectbox("Target column", targets)
        mode = st.segmented_control(
            "Run mode",
            options=[item.value for item in AutoRunMode],
            default=AutoRunMode.AUTO.value,
            format_func=lambda value: value.title(),
        )
        budget = st.number_input("Time budget (seconds)", min_value=10, value=120, step=10)
        model_name = st.text_input("Saved model name", value=f"{dataset_name}-autorun")
        submitted = st.form_submit_button("Review plan", type="primary")
    if submitted:
        st.session_state["autorun_config"] = AutoRunConfig(
            target_column=target,
            mode=AutoRunMode(mode),
            time_budget=int(budget),
            model_name=model_name,
        )
    config = st.session_state.get("autorun_config")
    if config is not None:
        plan = plan_auto_run(frame, config)
        st.subheader("Plan")
        st.write(
            f"**Task:** {plan.task_type.title()} · **Engine:** {plan.engine.upper()} · "
            f"**Metric:** {plan.primary_metric}"
        )
        for warning in plan.warnings:
            st.warning(warning)
        if st.button("Launch background training", type="primary", disabled=bool(plan.warnings)):
            service = BackgroundJobService(store=store, jobs_dir=state.settings.artifacts.root_dir / "jobs")
            record = service.submit_auto_run(frame, config)
            st.session_state["autorun_job_id"] = record.job_id
            st.rerun()
    _render_active_job(store)


@st.fragment(run_every="2s")
def _render_active_job(store) -> None:  # noqa: ANN001
    job_id = st.session_state.get("autorun_job_id")
    if not job_id:
        return
    job = store.get_job(job_id)
    if job is None:
        return
    st.subheader("Current job")
    progress = int(job.metadata.get("progress", 0))
    st.progress(progress, text=str(job.metadata.get("stage", job.status.value)).title())
    if job.status == AppJobStatus.RUNNING:
        if st.button("Cancel job"):
            BackgroundJobService(store=store, jobs_dir=Path(str(job.metadata["job_dir"])).parent).cancel(job_id)
            st.rerun()
    elif job.status == AppJobStatus.SUCCESS:
        st.success(f"Model ready: {job.primary_artifact_path}")
    elif job.status in {AppJobStatus.FAILED, AppJobStatus.CANCELLED}:
        st.error(str(job.metadata.get("error", job.status.value)))
