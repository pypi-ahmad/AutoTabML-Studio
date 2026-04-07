"""Dedicated Streamlit page for dataset intake and active selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.pages.dataset_workspace import get_active_loaded_dataset, go_to_page, render_dataset_workspace
from app.pages.ui_labels import render_metadata_table
from app.pages.workflow_guide import render_next_step_hint, render_workflow_banner
from app.state.session import get_or_init_state
from app.storage import build_metadata_store


def render_dataset_intake_page() -> None:
    state = get_or_init_state()
    metadata_store = build_metadata_store(state.settings)

    st.title("📥 Load Data")
    render_workflow_banner(current_step=1)
    st.write(
        "Load a dataset, review your cleaned data, and choose which dataset the rest of the app works with."
    )

    render_dataset_workspace(
        title="Load Your Data",
        caption=(
            "Load from an upload, local file path, or URL. The most recently loaded dataset is automatically selected as the current dataset."
        ),
        key_prefix="dataset_intake",
    )

    active_name, active_dataset = get_active_loaded_dataset(metadata_store=metadata_store)
    if active_dataset is None or active_name is None:
        st.info(
            "**Upload or connect a dataset above to get started.**\n\n"
            "Once loaded, you'll see a preview of your data, column details, "
            "and buttons to continue to the next step."
        )
        return

    dataframe = active_dataset.dataframe
    metadata = active_dataset.metadata

    from app.pages.ui_labels import DATASET_SOURCE_LABELS, format_enum_value
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Current dataset", active_name)
    metric_col2.metric("Rows", f"{len(dataframe):,}")
    metric_col3.metric("Columns", f"{len(dataframe.columns):,}")
    metric_col4.metric("Source", DATASET_SOURCE_LABELS.get(metadata.source_type.value, format_enum_value(metadata.source_type.value)))

    with st.expander("Dataset details", expanded=False):
        _md = metadata.model_dump(mode="json")
        render_metadata_table(_md)

    st.subheader("Data Preview")
    st.dataframe(active_dataset.preview(50), width="stretch")

    st.subheader("Column Summary")
    st.caption(
        "**Dtype** = data type · **Non-null** = rows with a value · "
        "**Null %** = percentage of missing values · **Unique** = number of distinct values"
    )
    st.dataframe(_build_schema_frame(dataframe), width="stretch")

    if metadata.normalization_actions:
        with st.expander("Data Cleanup Steps", expanded=True):
            for action in metadata.normalization_actions:
                st.markdown(f"- {action}")

    _render_uci_source_details(metadata.source_details)

    st.subheader("Continue")
    render_next_step_hint(current_step=1)

    st.caption("**Optional preparation** — check your data before modeling:")
    opt_col1, opt_col2, _ = st.columns([2, 2, 4])
    if opt_col1.button("✅ Validate Data", key="dataset_intake_to_validation", help="Check for missing values, duplicates, and other quality issues."):
        go_to_page("Validation")
    if opt_col2.button("📊 Explore Data", key="dataset_intake_to_profiling", help="Visual summary of distributions, correlations, and statistics."):
        go_to_page("Profiling")


def _build_schema_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    null_counts = dataframe.isna().sum()
    return pd.DataFrame(
        {
            "Column": dataframe.columns,
            "Dtype": [str(dtype) for dtype in dataframe.dtypes],
            "Non-null": [int(dataframe[column].notna().sum()) for column in dataframe.columns],
            "Nulls": [int(null_counts[column]) for column in dataframe.columns],
            "Null %": [round(float(null_counts[column]) / max(len(dataframe), 1) * 100, 2) for column in dataframe.columns],
            "Unique": [int(dataframe[column].nunique(dropna=True)) for column in dataframe.columns],
        }
    )


def _render_uci_source_details(source_details: dict[str, object]) -> None:
    if source_details.get("source_kind") != "uci_repo":
        return

    st.subheader("UCI Repository Details")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("UCI ID", source_details.get("uci_id") or "N/A")
    metric_col2.metric("Task", source_details.get("uci_task") or "N/A")
    metric_col3.metric("Area", source_details.get("uci_area") or "N/A")
    metric_col4.metric("Instances", source_details.get("uci_num_instances") or "N/A")

    additional_info = source_details.get("uci_additional_info")
    summary = None
    if isinstance(additional_info, dict):
        summary = additional_info.get("summary")
    if not summary:
        summary = source_details.get("uci_abstract")
    if summary:
        st.write(str(summary))

    columns_col1, columns_col2, columns_col3 = st.columns(3)
    columns_col1.caption(f"ID columns: {', '.join(source_details.get('id_columns', [])) or 'None'}")
    columns_col2.caption(f"Input columns: {', '.join(source_details.get('feature_columns', [])) or 'None'}")
    columns_col3.caption(f"Target columns: {', '.join(source_details.get('target_columns', [])) or 'None'}")

    variables = source_details.get("uci_variables")
    if isinstance(variables, list) and variables:
        st.markdown("**Variables**")
        st.dataframe(pd.DataFrame(variables), width="stretch")