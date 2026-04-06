"""Dashboard page – default workspace mode placeholder."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.pages.dataset_workspace import get_active_loaded_dataset, get_loaded_datasets, go_to_page
from app.state.session import get_or_init_state
from app.storage import build_metadata_store


def render_dashboard_page() -> None:
    state = get_or_init_state()
    metadata_store = build_metadata_store(state.settings)
    startup_status = st.session_state.get("startup_status")
    st.title("📊 Dashboard")
    st.write("Local-first workspace overview for recent datasets, jobs, and saved models.")
    st.caption(
        f"Provider: **{state.provider.value}** · "
        f"Backend: **{state.execution_backend.value}** · "
        f"Model: **{state.selected_model_id or '(none)'}**"
    )

    active_name, active_dataset = get_active_loaded_dataset(metadata_store=metadata_store)
    action_col, status_col = st.columns([2, 5])
    if action_col.button("Open Dataset Intake", key="dashboard_open_dataset_intake"):
        go_to_page("Dataset Intake")
    if active_dataset is None or active_name is None:
        status_col.info(
            "No active dataset selected. Open Dataset Intake to load one before validation, profiling, benchmark, or experiment workflows."
        )
    else:
        status_col.success(
            f"Active dataset: '{active_name}' · {len(active_dataset.dataframe):,} rows · {len(active_dataset.dataframe.columns):,} columns"
        )
        with st.expander("Active Dataset Preview", expanded=False):
            st.dataframe(active_dataset.preview(10), width="stretch")

    if startup_status and startup_status.issues:
        with st.expander("Startup Checks", expanded=bool(startup_status.errors)):
            for issue in startup_status.issues:
                if issue.severity == "error":
                    st.error(issue.message)
                elif issue.severity == "warning":
                    st.warning(issue.message)
                else:
                    st.info(issue.message)

    if metadata_store is None:
        st.info(
            "Local metadata store is not available. "
            "Run `autotabml init-local-storage` to initialize the workspace database."
        )
        return

    jobs = metadata_store.list_recent_jobs(limit=8)
    datasets = metadata_store.list_recent_datasets(limit=8)
    models = metadata_store.list_saved_local_models(limit=8)

    loaded = get_loaded_datasets()
    job_col, dataset_col, model_col, session_col = st.columns(4)
    job_col.metric("Recent jobs", len(jobs))
    dataset_col.metric("Tracked datasets", len(datasets))
    model_col.metric("Saved local models", len(models))
    session_col.metric("Session datasets", len(loaded))

    if jobs:
        st.subheader("Recent Local Jobs")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Updated": job.updated_at,
                        "Type": job.job_type.value,
                        "Status": job.status.value,
                        "Dataset": job.dataset_name or "",
                        "Title": job.title or "",
                        "Artifact": str(job.primary_artifact_path) if job.primary_artifact_path else "",
                    }
                    for job in jobs
                ]
            ),
            width="stretch",
        )
    else:
        st.info("No local jobs have been recorded yet. Run validation, profiling, benchmarking, experiments, or prediction to populate the workspace history.")

    dataset_section, model_section = st.columns(2)
    with dataset_section:
        st.subheader("Tracked Datasets")
        if datasets:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Name": dataset.display_name or dataset.source_locator,
                            "Rows": dataset.row_count,
                            "Columns": dataset.column_count,
                            "Source": dataset.source_type,
                        }
                        for dataset in datasets
                    ]
                ),
                width="stretch",
            )
        else:
            st.caption("Datasets are recorded when local workflows run against them.")

    with model_section:
        st.subheader("Saved Local Models")
        if models:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Model": model.model_name,
                            "Task": model.task_type,
                            "Target": model.target_column or "",
                            "Path": str(model.model_path),
                        }
                        for model in models
                    ]
                ),
                width="stretch",
            )
        else:
            st.caption("Finalize and save an experiment model to track it here.")
