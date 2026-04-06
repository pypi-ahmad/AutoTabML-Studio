"""Streamlit page for the run history center."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.state.session import get_or_init_state
from app.tracking.filters import RunHistoryFilter, RunHistorySort, RunSortField, SortDirection
from app.tracking.mlflow_query import is_mlflow_available
from app.tracking.schemas import RunStatus, RunType
from app.security.masking import safe_error_message


def render_history_page() -> None:
    state = get_or_init_state()
    settings = state.settings.tracking
    st.title("📋 Run History")

    if not is_mlflow_available():
        st.warning("mlflow is not installed. Install with: `pip install mlflow`")
        return

    from app.tracking.history_service import HistoryService

    service = HistoryService(
        tracking_uri=settings.tracking_uri,
        default_experiment_names=settings.default_experiment_names,
        default_limit=settings.history_page_default_limit,
    )

    # --- Filter controls ---
    with st.expander("Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            run_type_value = st.selectbox(
                "Run type",
                options=["all", "benchmark", "experiment", "unknown"],
                index=0,
                key="hist_run_type",
            )
        with col2:
            task_type_value = st.text_input(
                "Task type",
                value="",
                key="hist_task_type",
            ).strip()
        with col3:
            status_value = st.selectbox(
                "Status",
                options=["all", "FINISHED", "RUNNING", "FAILED", "KILLED"],
                index=0,
                key="hist_status",
            )

        col4, col5 = st.columns(2)
        with col4:
            target_column = st.text_input("Target column", value="", key="hist_target").strip()
        with col5:
            model_name = st.text_input("Model name (contains)", value="", key="hist_model").strip()

        experiment_names_raw = st.text_input(
            "Experiment names (comma-separated, blank for defaults)",
            value="",
            key="hist_experiments",
        ).strip()

    sort_col, dir_col, limit_col = st.columns(3)
    with sort_col:
        sort_field = RunSortField(
            st.selectbox(
                "Sort by",
                options=[f.value for f in RunSortField],
                index=0,
                key="hist_sort",
            )
        )
    with dir_col:
        sort_dir = SortDirection(
            st.selectbox(
                "Direction",
                options=[d.value for d in SortDirection],
                index=0,
                key="hist_sort_dir",
            )
        )
    with limit_col:
        limit = int(
            st.number_input(
                "Max results",
                min_value=1,
                max_value=500,
                value=settings.history_page_default_limit,
                step=10,
                key="hist_limit",
            )
        )

    # --- Build filter ---
    history_filter = RunHistoryFilter(
        experiment_names=(
            [n.strip() for n in experiment_names_raw.split(",") if n.strip()]
            if experiment_names_raw
            else []
        ),
        run_type=RunType(run_type_value) if run_type_value != "all" else None,
        task_type=task_type_value or None,
        target_column=target_column or None,
        model_name=model_name or None,
        status=RunStatus(status_value) if status_value != "all" else None,
    )
    sort = RunHistorySort(field=sort_field, direction=sort_dir)

    # --- Fetch and display ---
    try:
        runs = service.list_runs(history_filter=history_filter, sort=sort, limit=limit)
    except Exception as exc:
        st.error(f"Failed to query run history: {safe_error_message(exc)}")
        return

    if not runs:
        st.info("No runs found matching the current filters.")
        return

    st.caption(f"Showing **{len(runs)}** run(s)")

    rows = []
    for run in runs:
        rows.append({
            "Run ID": run.run_id[:12],
            "Type": run.run_type.value,
            "Status": run.status.value,
            "Task": run.task_type or "",
            "Model": run.model_name or "",
            "Score": run.primary_metric_value,
            "Metric": run.primary_metric_name or "",
            "Duration (s)": run.duration_seconds,
            "Start": run.start_time,
            "Experiment": run.experiment_name or "",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch")

    # --- Run detail view ---
    run_ids = [r.run_id for r in runs]
    run_labels = [
        f"{r.run_id[:12]} | {r.run_name or r.run_type.value}"
        for r in runs
    ]
    selected_label = st.selectbox("Inspect run", options=run_labels, key="hist_inspect")
    if selected_label:
        selected_index = run_labels.index(selected_label)
        selected_run_id = run_ids[selected_index]

        try:
            detail = service.get_run_detail(selected_run_id)
        except Exception as exc:
            st.error(f"Could not fetch run detail: {safe_error_message(exc)}")
            return

        st.subheader("Run Detail")
        st.caption(f"Run ID: `{detail.run_id}`")
        st.caption(f"Artifact URI: `{detail.artifact_uri or 'N/A'}`")

        detail_col1, detail_col2, detail_col3 = st.columns(3)
        detail_col1.metric("Type", detail.run_type.value)
        detail_col2.metric("Status", detail.status.value)
        detail_col3.metric("Duration (s)", f"{detail.duration_seconds:.1f}" if detail.duration_seconds else "N/A")

        if detail.params:
            with st.expander("Parameters"):
                st.json(detail.params)

        if detail.metrics:
            with st.expander("Metrics"):
                st.json({k: v for k, v in detail.metrics.items()})

        if detail.tags:
            with st.expander("Tags"):
                st.json(detail.tags)

        if detail.artifact_paths:
            with st.expander("Artifacts"):
                st.caption(
                    "Artifact paths are shown for manual copy/reference. Direct download is not implemented yet."
                )
                artifact_rows = []
                for path in detail.artifact_paths:
                    artifact_rows.append({
                        "Path": path,
                        "Source URI": _artifact_source_uri(detail.artifact_uri, path),
                    })
                st.dataframe(pd.DataFrame(artifact_rows), width="stretch")


def _artifact_source_uri(artifact_uri: str | None, artifact_path: str) -> str:
    if "://" in artifact_path:
        return artifact_path
    if not artifact_uri:
        return artifact_path
    return f"{artifact_uri.rstrip('/')}/{artifact_path.lstrip('/')}"
