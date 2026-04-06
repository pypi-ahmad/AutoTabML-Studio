"""Dedicated Streamlit page for dataset intake and active selection."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.pages.dataset_workspace import get_active_loaded_dataset, go_to_page, render_dataset_workspace
from app.state.session import get_or_init_state
from app.storage import build_metadata_store


def render_dataset_intake_page() -> None:
    state = get_or_init_state()
    metadata_store = build_metadata_store(state.settings)

    st.title("🧾 Dataset Intake")
    st.write(
        "Load a dataset into the current session, inspect the normalized dataframe, and set the active dataset used by validation, profiling, benchmark, and experiment workflows."
    )

    render_dataset_workspace(
        title="Load Dataset Sources",
        caption=(
            "Load from an upload, local file path, or URL. Newly loaded datasets become active automatically, "
            "and the active selection is also persisted into the local workspace metadata store."
        ),
        key_prefix="dataset_intake",
    )

    active_name, active_dataset = get_active_loaded_dataset(metadata_store=metadata_store)
    if active_dataset is None or active_name is None:
        st.info(
            "Load a dataset above to preview the normalized dataframe, inspect schema and metadata, and continue into downstream workflows."
        )
        return

    dataframe = active_dataset.dataframe
    metadata = active_dataset.metadata

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Active dataset", active_name)
    metric_col2.metric("Rows", f"{len(dataframe):,}")
    metric_col3.metric("Columns", f"{len(dataframe.columns):,}")
    metric_col4.metric("Source", metadata.source_type.value)

    st.caption(f"Source locator: **{metadata.source_locator}**")

    st.subheader("Normalized Preview")
    st.dataframe(active_dataset.preview(50), width="stretch")

    st.subheader("Schema")
    st.dataframe(_build_schema_frame(dataframe), width="stretch")

    if metadata.normalization_actions:
        with st.expander("Normalization Actions", expanded=True):
            for action in metadata.normalization_actions:
                st.markdown(f"- {action}")

    _render_uci_source_details(metadata.source_details)

    with st.expander("Dataset Metadata", expanded=False):
        st.json(metadata.model_dump(mode="json"))

    st.subheader("Continue")
    validation_col, profiling_col, benchmark_col, experiment_col = st.columns(4)
    if validation_col.button("Validation", key="dataset_intake_to_validation"):
        go_to_page("Validation")
    if profiling_col.button("Profiling", key="dataset_intake_to_profiling"):
        go_to_page("Profiling")
    if benchmark_col.button("Benchmark", key="dataset_intake_to_benchmark"):
        go_to_page("Benchmark")
    if experiment_col.button("Experiment", key="dataset_intake_to_experiment"):
        go_to_page("Experiment")


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
    columns_col2.caption(f"Feature columns: {', '.join(source_details.get('feature_columns', [])) or 'None'}")
    columns_col3.caption(f"Target columns: {', '.join(source_details.get('target_columns', [])) or 'None'}")

    variables = source_details.get("uci_variables")
    if isinstance(variables, list) and variables:
        st.markdown("**Variables**")
        st.dataframe(pd.DataFrame(variables), width="stretch")