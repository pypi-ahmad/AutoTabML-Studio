"""Streamlit validation page – run and view data validation results."""

from __future__ import annotations

import streamlit as st

from app.pages.dataset_workspace import go_to_page, render_dataset_header
from app.pages.ui_cache import get_metadata_store
from app.pages.ui_labels import render_metadata_table
from app.pages.workflow_guide import render_workflow_banner
from app.state.session import get_or_init_state


def render_validation_page() -> None:
    state = get_or_init_state()
    st.title("✅ Data Validation")
    render_workflow_banner(current_step=2)
    st.caption(
        "`Optional step` — Check your data for common quality problems before training a model. "
        "You can skip this and go straight to **Find Best Model** if you prefer."
    )
    settings = state.settings.validation
    metadata_store = get_metadata_store(state.settings)

    selected_name, loaded_dataset = render_dataset_header("Validation", key_prefix="validation", metadata_store=metadata_store)
    if selected_name is None or loaded_dataset is None:
        return

    df = loaded_dataset.dataframe

    st.caption(f"Rows: **{len(df)}** · Columns: **{len(df.columns)}**")

    # --- Target column ---
    columns = ["(none)"] + list(df.columns)
    target_col = st.selectbox(
        "Target column (optional)",
        columns,
        key="val_target",
        help="The column you want to predict. Validation will run extra checks on this column (e.g. missing values, data type).",
    )
    target = target_col if target_col != "(none)" else None

    # --- Advanced options ---
    with st.expander("Advanced options", expanded=False):
        required_cols_raw = st.text_input(
            "Required columns (comma-separated)",
            key="val_required_cols",
            help="Columns that MUST be present in the dataset. The check fails if any are missing.",
        )
        required_cols = [c.strip() for c in required_cols_raw.split(",") if c.strip()] if required_cols_raw else []

        uniqueness_cols_raw = st.text_input(
            "Columns that should be unique (comma-separated)",
            key="val_uniqueness_cols",
            help="Columns where every value should be different (e.g. an ID column). Duplicates will be flagged.",
        )
        uniqueness_cols = [c.strip() for c in uniqueness_cols_raw.split(",") if c.strip()] if uniqueness_cols_raw else []

        enable_leakage = st.checkbox(
            "Check for data leakage",
            value=True,
            key="val_leakage",
            help="Detect columns that accidentally reveal the answer (e.g. a column that is derived from the target).",
        )
        min_rows = st.number_input(
            "Minimum row count",
            value=int(settings.min_row_threshold),
            min_value=0,
            key="val_min_rows",
            help="Fail validation if the dataset has fewer rows than this number.",
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
        with st.spinner("Checking data quality…"):
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
        st.info(
            "**Ready to validate.**\n\n"
            "Click **Run Validation** above to check your data for quality issues.\n\n"
            "**What happens:** Automated checks look for missing values, duplicate rows, data-type problems, "
            "and potential data leakage. You'll get a pass/warn/fail report with clear next steps."
        )
        return

    _render_summary(summary, selected_name)

    # Navigation shortcuts
    st.divider()
    c1, c2, _ = st.columns([2, 2, 4])
    c1.markdown("**Ready to model?**")
    if c2.button("🏁 Go to Find Best Model", key="val_goto_benchmark", type="primary"):
        go_to_page("Quick Benchmark")


def _render_summary(summary, selected_name: str) -> None:  # noqa: ANN001
    """Render validation summary cards and check details."""
    # Plain-English summary
    _total = summary.total_checks
    if summary.failed_count == 0 and summary.warning_count == 0:
        _verdict = "Everything looks great — all checks passed with no issues."
        _next = "Your data is ready. Head to **Find Best Model** to benchmark algorithms."
    elif summary.failed_count == 0:
        _verdict = f"{summary.warning_count} minor issue(s) flagged, but nothing blocking."
        _next = "Review the warnings below, then proceed to **Find Best Model**."
    else:
        _verdict = f"{summary.failed_count} check(s) failed — the data may need cleaning before modeling."
        _next = "Expand the check details below, fix the issues in your source data, and re-run."
    st.info(
        f"**What happened:** Ran **{_total}** data quality checks — "
        f"**{summary.passed_count}** passed, **{summary.warning_count}** warnings, "
        f"**{summary.failed_count}** failed.\n\n"
        f"**Looks good?** {_verdict}\n\n"
        f"**Next step:** {_next}"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Passed", summary.passed_count)
    col2.metric("Warnings", summary.warning_count)
    col3.metric("Failed", summary.failed_count)

    st.markdown(f"**Total checks:** {summary.total_checks}")

    if summary.checks:
        with st.expander("Check details", expanded=summary.has_failures):
            for check in summary.checks:
                icon = "✅" if check.passed else ("⚠️" if check.severity.value == "warning" else "❌")
                st.markdown(f"{icon} **{check.check_name}** — {check.message}")
                if check.details:
                    _detail_items = check.details if isinstance(check.details, dict) else {"details": check.details}
                    for _dk, _dv in _detail_items.items():
                        st.markdown(f"- **{_dk}:** {_dv}")
                    with st.expander("Full details"):
                        render_metadata_table(_detail_items, show_hidden=True)

    bundle = st.session_state.get("validation_bundles", {}).get(selected_name)
    if bundle:
        with st.expander("Reports & Downloads"):
            for _label, _path in [
                ("JSON summary", bundle.summary_json_path),
                ("Markdown report", bundle.markdown_report_path),
                ("GX Data Docs", bundle.gx_data_docs_path),
            ]:
                if _path is None:
                    continue
                if _path.exists():
                    st.download_button(
                        label=f"Download {_label}",
                        data=_path.read_bytes(),
                        file_name=_path.name,
                        key=f"val_dl_{_label}_{_path.name}",
                    )
