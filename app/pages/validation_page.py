"""Streamlit validation page – run and view data validation results."""

from __future__ import annotations

import streamlit as st

from app.pages.dataset_workspace import (
    get_active_loaded_dataset,
    render_active_dataset_banner,
    render_dataset_gateway_notice,
)
from app.state.session import get_or_init_state
from app.storage import build_metadata_store


def render_validation_page() -> None:
    state = get_or_init_state()
    st.title("✅ Data Validation")
    settings = state.settings.validation
    metadata_store = build_metadata_store(state.settings)

    selected_name, loaded_dataset = get_active_loaded_dataset(metadata_store=metadata_store)
    if selected_name is None or loaded_dataset is None:
        render_dataset_gateway_notice("Validation", key_prefix="validation")
        return

    render_active_dataset_banner(selected_name, key_prefix="validation")
    df = loaded_dataset.dataframe

    st.caption(f"Rows: **{len(df)}** · Columns: **{len(df.columns)}**")

    # --- Target column ---
    columns = ["(none)"] + list(df.columns)
    target_col = st.selectbox("Target column (optional)", columns, key="val_target")
    target = target_col if target_col != "(none)" else None

    # --- Advanced options ---
    with st.expander("Advanced options"):
        required_cols_raw = st.text_input(
            "Required columns (comma-separated)", key="val_required_cols"
        )
        required_cols = [c.strip() for c in required_cols_raw.split(",") if c.strip()] if required_cols_raw else []

        uniqueness_cols_raw = st.text_input(
            "Uniqueness check columns (comma-separated)", key="val_uniqueness_cols"
        )
        uniqueness_cols = [c.strip() for c in uniqueness_cols_raw.split(",") if c.strip()] if uniqueness_cols_raw else []

        enable_leakage = st.checkbox("Enable leakage heuristics", value=True, key="val_leakage")
        min_rows = st.number_input(
            "Minimum row count",
            value=int(settings.min_row_threshold),
            min_value=0,
            key="val_min_rows",
        )

    # --- Run validation ---
    if st.button("Run Validation", key="val_run"):
        from app.validation.schemas import ValidationRuleConfig
        from app.validation.service import validate_dataset

        config = ValidationRuleConfig(
            target_column=target,
            required_columns=required_cols,
            uniqueness_columns=uniqueness_cols,
            enable_leakage_heuristics=enable_leakage,
            min_row_count=min_rows,
            null_warn_pct=settings.null_warn_pct,
            null_fail_pct=settings.null_fail_pct,
        )

        artifacts_dir = settings.artifacts_dir
        summary, bundle = validate_dataset(
            df,
            config,
            dataset_name=selected_name,
            artifacts_dir=artifacts_dir,
            loaded_dataset=loaded_dataset,
            metadata_store=metadata_store,
        )
        summaries = st.session_state.setdefault("validation_summaries", {})
        bundles = st.session_state.setdefault("validation_bundles", {})
        summaries[selected_name] = summary
        bundles[selected_name] = bundle
        st.success("Validation complete.")

    # --- Display results ---
    summary = st.session_state.get("validation_summaries", {}).get(selected_name)
    if summary is None:
        st.caption("Run validation to see results.")
        return

    _render_summary(summary, selected_name)


def _render_summary(summary, selected_name: str) -> None:  # noqa: ANN001
    """Render validation summary cards and check details."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Passed", summary.passed_count)
    col2.metric("Warnings", summary.warning_count)
    col3.metric("Failed", summary.failed_count)

    st.markdown(f"**Total checks:** {summary.total_checks}")

    if summary.checks:
        with st.expander("Check details", expanded=summary.has_failures):
            for check in summary.checks:
                icon = "✅" if check.passed else ("⚠️" if check.severity.value == "warning" else "❌")
                st.markdown(f"{icon} **{check.check_name}** [{check.severity.value}] — {check.message}")
                if check.details:
                    st.json(check.details)

    bundle = st.session_state.get("validation_bundles", {}).get(selected_name)
    if bundle:
        with st.expander("Artifacts"):
            if bundle.summary_json_path:
                st.markdown(f"JSON summary: `{bundle.summary_json_path}`")
            if bundle.markdown_report_path:
                st.markdown(f"Markdown report: `{bundle.markdown_report_path}`")
            if bundle.gx_data_docs_path:
                st.markdown(f"GX Data Docs: `{bundle.gx_data_docs_path}`")
