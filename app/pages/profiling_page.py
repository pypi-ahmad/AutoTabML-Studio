"""Streamlit profiling / EDA page – run and view dataset profiling results."""

from __future__ import annotations

import streamlit as st

from app.pages.dataset_workspace import render_dataset_header
from app.security.masking import safe_error_message
from app.state.session import get_or_init_state
from app.storage import build_metadata_store


def render_profiling_page() -> None:
    state = get_or_init_state()
    st.title("📊 EDA / Profiling")
    prof_settings = state.settings.profiling
    metadata_store = build_metadata_store(state.settings)

    selected_name, loaded_dataset = render_dataset_header("Profiling", key_prefix="profiling", metadata_store=metadata_store)
    if selected_name is None or loaded_dataset is None:
        return

    df = loaded_dataset.dataframe
    n_rows, n_cols = df.shape

    st.caption(f"Rows: **{n_rows}** · Columns: **{n_cols}**")

    # Large-dataset warning
    if n_rows > prof_settings.large_dataset_row_threshold or n_cols > prof_settings.large_dataset_col_threshold:
        st.warning(
            f"This dataset ({n_rows} rows, {n_cols} cols) exceeds the large-dataset threshold. "
            "Profiling will run in **minimal** mode and may use **sampling** to limit resource usage."
        )

    # --- Run profiling ---
    if st.button("Generate Profile", key="prof_run"):
        from app.profiling.ydata_runner import is_ydata_available, profiling_install_guidance

        if not is_ydata_available():
            st.error(profiling_install_guidance())
            return

        from app.profiling.service import profile_dataset

        with st.spinner("Generating profile report…"):
            try:
                summary, bundle = profile_dataset(
                    df,
                    mode=prof_settings.default_mode,
                    dataset_name=selected_name,
                    artifacts_dir=prof_settings.artifacts_dir,
                    loaded_dataset=loaded_dataset,
                    metadata_store=metadata_store,
                    large_dataset_row_threshold=prof_settings.large_dataset_row_threshold,
                    large_dataset_col_threshold=prof_settings.large_dataset_col_threshold,
                    sampling_row_threshold=prof_settings.sampling_row_threshold,
                    sample_size=prof_settings.sample_size,
                )
                summaries = st.session_state.setdefault("profiling_summaries", {})
                bundles = st.session_state.setdefault("profiling_bundles", {})
                summaries[selected_name] = summary
                bundles[selected_name] = bundle
                st.success("Profiling complete.")
            except Exception as exc:
                st.error(f"Profiling failed: {safe_error_message(exc)}")
                return

    # --- Display results ---
    summary = st.session_state.get("profiling_summaries", {}).get(selected_name)
    if summary is None:
        st.caption("Run profiling to see results.")
        return

    _render_summary(summary, selected_name)


def _render_summary(summary, selected_name: str) -> None:  # noqa: ANN001
    """Render profiling summary cards."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{summary.row_count:,}")
    col2.metric("Columns", summary.column_count)
    col3.metric("Missing %", f"{summary.missing_cells_pct:.1f}%")
    col4.metric("Duplicates", summary.duplicate_row_count)

    st.markdown(
        f"**Numeric columns:** {summary.numeric_column_count} · "
        f"**Categorical columns:** {summary.categorical_column_count}"
    )

    if summary.memory_bytes:
        mb = summary.memory_bytes / (1024 * 1024)
        st.caption(f"Memory usage: {mb:.1f} MB")

    if summary.sampling_applied:
        st.info(f"Sampling was applied: {summary.sample_size_used:,} rows used for profiling.")

    st.caption(f"Report mode: **{summary.report_mode.value}**")

    if summary.high_cardinality_columns:
        st.warning(f"High-cardinality columns: {', '.join(summary.high_cardinality_columns)}")

    bundle = st.session_state.get("profiling_bundles", {}).get(selected_name)
    if bundle:
        with st.expander("Artifacts"):
            if bundle.html_report_path:
                st.markdown(f"HTML report: `{bundle.html_report_path}`")
            if bundle.summary_json_path:
                st.markdown(f"JSON summary: `{bundle.summary_json_path}`")
