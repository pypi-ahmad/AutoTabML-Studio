"""Streamlit profiling / EDA page – run and view dataset profiling results."""

from __future__ import annotations

import streamlit as st

from app.pages.dataset_workspace import go_to_page, render_dataset_header
from app.pages.shared_history import render_past_runs_section, render_saved_models_section
from app.pages.ui_cache import get_metadata_store
from app.pages.ui_errors import log_ui_exception
from app.pages.ui_labels import format_enum_value
from app.pages.workflow_guide import render_workflow_banner
from app.security.masking import safe_error_message
from app.state.session import get_or_init_state
from app.storage.models import AppJobType


def render_profiling_page() -> None:
    state = get_or_init_state()
    st.title("📊 Data Profiling")
    render_workflow_banner(current_step=2)
    st.caption(
        "`Optional step` — Generate a visual summary of your data (distributions, correlations, missing values) "
        "to get familiar with your dataset before modeling. You can skip this."
    )
    prof_settings = state.settings.profiling
    metadata_store = get_metadata_store(state.settings)

    selected_name, loaded_dataset = render_dataset_header("Profiling", key_prefix="profiling", metadata_store=metadata_store)
    if selected_name is None or loaded_dataset is None:
        return

    df = loaded_dataset.dataframe
    n_rows, n_cols = df.shape

    st.caption(f"Rows: **{n_rows}** · Columns: **{n_cols}**")

    # Large-dataset warning
    if n_rows > prof_settings.large_dataset_row_threshold or n_cols > prof_settings.large_dataset_col_threshold:
        st.warning(
            f"This is a large dataset ({n_rows:,} rows, {n_cols} columns). "
            "The report will use a **compact mode** and may analyse a random sample "
            "instead of every row to keep things fast."
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
                log_ui_exception(exc, operation="profiling.run")
                st.error(f"Profiling failed: {safe_error_message(exc)}")
                return

    # --- Display results ---
    summary = st.session_state.get("profiling_summaries", {}).get(selected_name)
    if summary is None:
        st.info(
            "**Ready to profile.**\n\n"
            "Click **Generate Profile** above.\n\n"
            "**What happens:** A detailed visual report is generated showing distributions, correlations, missing values, "
            "and outliers for every column. You can download it as HTML to share with your team."
        )
    else:
        _render_summary(summary, selected_name)

        # Navigation shortcuts
        st.divider()
        c1, c2, c3, _ = st.columns([2, 2, 2, 2])
        c1.markdown("**Ready to continue?**")
        if c2.button("✅ Validate Data", key="prof_goto_validation"):
            go_to_page("Validation")
        if c3.button("🏁 Find Best Model", key="prof_goto_benchmark", type="primary"):
            go_to_page("Quick Benchmark")

    # ── Past runs & saved models ───────────────────────────────────────
    st.divider()
    render_past_runs_section(metadata_store, AppJobType.PROFILING, key_prefix="prof")
    render_saved_models_section(state.settings.prediction, key_prefix="prof")


def _render_summary(summary, selected_name: str) -> None:  # noqa: ANN001
    """Render profiling summary cards."""
    # Plain-English summary
    _issues = []
    if summary.missing_cells_pct > 5:
        _issues.append(f"{summary.missing_cells_pct:.1f}% missing values")
    if summary.duplicate_row_count > 0:
        _issues.append(f"{summary.duplicate_row_count:,} duplicate rows")
    if summary.high_cardinality_columns:
        _issues.append(f"{len(summary.high_cardinality_columns)} high-cardinality column(s)")
    if _issues:
        _verdict = "Some things to watch: " + ", ".join(_issues) + "."
    else:
        _verdict = "The dataset looks clean — no major issues detected."
    st.info(
        f"**What happened:** Profiled **{summary.row_count:,}** rows across "
        f"**{summary.column_count}** columns "
        f"(**{summary.numeric_column_count}** numeric, **{summary.categorical_column_count}** categorical).\n\n"
        f"**Looks good?** {_verdict}\n\n"
        "**Next step:** Run **Validation** to check data quality, or jump straight to **Find Best Model**."
    )

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

    st.caption(f"Report mode: **{format_enum_value(summary.report_mode.value)}**")

    if summary.high_cardinality_columns:
        st.warning(f"High-cardinality columns: {', '.join(summary.high_cardinality_columns)}")

    bundle = st.session_state.get("profiling_bundles", {}).get(selected_name)
    if bundle:
        with st.expander("Reports & Downloads"):
            for _label, _path in [
                ("HTML report", bundle.html_report_path),
                ("JSON summary", bundle.summary_json_path),
            ]:
                if _path is None:
                    continue
                if _path.exists():
                    st.download_button(
                        label=f"Download {_label}",
                        data=_path.read_bytes(),
                        file_name=_path.name,
                        key=f"prof_dl_{_label}_{_path.name}",
                    )
