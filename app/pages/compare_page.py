"""Streamlit page for the side-by-side run comparison center."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.state.session import get_or_init_state
from app.tracking.mlflow_query import is_mlflow_available
from app.security.masking import safe_error_message


def render_compare_page() -> None:
    state = get_or_init_state()
    settings = state.settings.tracking
    st.title("⚖️ Run Comparison")

    if not is_mlflow_available():
        st.warning("mlflow is not installed. Install with: `pip install mlflow`")
        return

    from app.tracking.compare_service import ComparisonService
    from app.tracking.history_service import HistoryService

    history = HistoryService(
        tracking_uri=settings.tracking_uri,
        default_experiment_names=settings.default_experiment_names,
        default_limit=settings.history_page_default_limit,
    )
    comparison = ComparisonService()

    try:
        runs = history.list_runs(limit=settings.history_page_default_limit)
    except Exception as exc:
        st.error(f"Failed to query runs: {safe_error_message(exc)}")
        return

    if len(runs) < 2:
        st.info("At least two MLflow runs are required for comparison.")
        return

    run_labels = [
        f"{r.run_id[:12]} | {r.run_name or r.run_type.value} | {r.model_name or 'N/A'}"
        for r in runs
    ]

    col1, col2 = st.columns(2)
    with col1:
        left_label = st.selectbox("Left run", options=run_labels, index=0, key="cmp_left")
    with col2:
        right_default = 1 if len(run_labels) > 1 else 0
        right_label = st.selectbox("Right run", options=run_labels, index=right_default, key="cmp_right")

    left_run = runs[run_labels.index(left_label)]
    right_run = runs[run_labels.index(right_label)]

    if left_run.run_id == right_run.run_id:
        st.warning("Select two different runs to compare.")
        return

    bundle = comparison.compare(left_run, right_run)

    # --- Summary cards ---
    st.subheader("Comparison Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric(
        "Left",
        left_run.model_name or left_run.run_type.value,
        left_run.primary_metric_value,
    )
    summary_col2.metric(
        "Right",
        right_run.model_name or right_run.run_type.value,
        right_run.primary_metric_value,
    )
    summary_col3.metric("Comparable", "Yes" if bundle.comparable else "No")

    # --- Warnings ---
    if bundle.warnings:
        for warning in bundle.warnings:
            st.warning(warning)

    # --- Metrics deltas ---
    if bundle.metric_deltas:
        st.subheader("Metric Deltas")
        metric_rows = []
        for delta in bundle.metric_deltas:
            metric_rows.append({
                "Metric": delta.name,
                "Left": delta.left_value,
                "Right": delta.right_value,
                "Delta": delta.delta,
                "Better": delta.better_side or "",
            })
        st.dataframe(pd.DataFrame(metric_rows), width="stretch")

    # --- Config differences ---
    if bundle.config_differences:
        st.subheader("Configuration Differences")
        config_rows = []
        for diff in bundle.config_differences:
            config_rows.append({
                "Key": diff.key,
                "Left": diff.left_value or "N/A",
                "Right": diff.right_value or "N/A",
                "Category": diff.category,
            })
        st.dataframe(pd.DataFrame(config_rows), width="stretch")

    # --- Artifact comparison ---
    st.subheader("Artifact Availability")
    artifact_col1, artifact_col2 = st.columns(2)
    with artifact_col1:
        st.markdown(f"**Left** artifact URI:")
        st.caption(f"`{left_run.artifact_uri or 'N/A'}`")
    with artifact_col2:
        st.markdown(f"**Right** artifact URI:")
        st.caption(f"`{right_run.artifact_uri or 'N/A'}`")
    st.caption("Artifact URIs are references only. Use Run History to inspect artifact paths; direct download is not implemented yet.")

    # --- Save comparison ---
    if st.button("Save Comparison Artifacts", key="cmp_save"):
        from app.tracking.artifacts import write_comparison_artifacts

        try:
            paths = write_comparison_artifacts(bundle, settings.comparison_artifacts_dir)
            st.success("Comparison artifacts saved.")
            for label, path in paths.items():
                st.markdown(f"- {label}: `{path}`")
        except Exception as exc:
            st.error(f"Failed to save comparison artifacts: {safe_error_message(exc)}")
