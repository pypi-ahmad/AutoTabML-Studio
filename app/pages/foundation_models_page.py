"""Streamlit workflow for Google TabFM and TimesFM foundation models."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from uuid import uuid4

import altair as alt
import streamlit as st

from app.errors import log_exception
from app.modeling.foundation import TabFMConfig, TabFMService, TimesFMConfig, TimesFMService
from app.modeling.foundation.tracking import log_foundation_run
from app.pages.dataset_workspace import render_dataset_header
from app.pages.ui_cache import get_metadata_store
from app.security.masking import safe_error_message
from app.state.session import get_or_init_state
from app.storage import AppJobStatus, AppJobType, JobRecord, SavedLocalModelRecord, ensure_dataset_record


@st.cache_resource
def _tabfm_service() -> TabFMService:
    return TabFMService()


@st.cache_resource
def _timesfm_service() -> TimesFMService:
    return TimesFMService()


def render_foundation_models_page() -> None:
    """Render a single conditional page for TabFM and TimesFM."""

    state = get_or_init_state()
    store = get_metadata_store(state.settings)
    st.title("Foundation Models")
    st.caption("Local Google foundation models for tabular evaluation and time-series forecasting.")
    if str(state.execution_backend).lower().endswith("colab"):
        st.info("Foundation models run on this machine only; the Colab execution backend is not used.")

    model_family = st.segmented_control(
        "Workflow",
        ["Tabular", "Time Series"],
        default="Tabular",
        key="foundation_workflow",
    )
    selected_name, loaded = render_dataset_header(
        "Foundation Models",
        key_prefix="foundation",
        metadata_store=store,
    )
    if selected_name is None or loaded is None:
        return
    ensure_dataset_record(store, loaded, dataset_name=selected_name)
    if model_family == "Time Series":
        _render_timesfm(selected_name, loaded, state.settings, store)
    else:
        _render_tabfm(selected_name, loaded, state.settings, store)


def _render_tabfm(dataset_name, loaded, settings, store) -> None:  # noqa: ANN001
    dataframe = loaded.dataframe
    st.subheader("TabFM 1.0")
    st.warning(
        "Research use only. The pretrained weights use the tabfm-non-commercial-v1.0 license and cannot be used "
        "commercially or in production. Saved contexts are prediction-only and cannot be registered or deployed."
    )
    _dependency_status("tabfm", "uv sync --extra tabfm")
    with st.form("tabfm_run_form"):
        target = st.selectbox("Target column", list(dataframe.columns))
        task_type = st.selectbox("Task type", ["auto", "classification", "regression"])
        left, right = st.columns(2)
        context_rows = int(left.number_input("Context rows", min_value=2, max_value=10000, value=100))
        n_estimators = int(right.number_input("Ensemble members", min_value=1, max_value=128, value=32))
        save_context = st.checkbox("Save sampled research context for Predictions", value=False)
        context_name = st.text_input("Saved context name", value=f"{dataset_name}-tabfm", disabled=not save_context)
        accept_license = st.checkbox("I accept the non-commercial, non-production weights license")
        allow_download = st.checkbox("Allow the first pinned model download (about 1 GB)")
        submitted = st.form_submit_button("Run TabFM evaluation", type="primary")

    if submitted:
        config = TabFMConfig(
            target_column=target,
            task_type=task_type,
            context_rows=context_rows,
            n_estimators=n_estimators,
        )
        with st.spinner("Running pinned TabFM evaluation…"):
            try:
                result = _tabfm_service().run(
                    dataframe,
                    config,
                    accept_license=accept_license,
                    allow_download=allow_download,
                    output_dir=settings.artifacts.experiments_dir / "foundation",
                )
                if save_context:
                    saved = _tabfm_service().save_context(
                        result,
                        name=context_name,
                        output_dir=settings.artifacts.models_dir,
                    )
                    if store is not None:
                        store.upsert_saved_local_model(
                            SavedLocalModelRecord(
                                record_id=f"tabfm-{uuid4().hex}",
                                model_name=context_name,
                                model_path=saved.context_path,
                                task_type=result.task_type,
                                target_column=target,
                                dataset_fingerprint=loaded.metadata.content_hash or loaded.metadata.schema_hash,
                                metadata_path=saved.metadata_path,
                                metadata={"framework": "tabfm", "research_only": True, "deployable": False},
                            )
                        )
                run_id, warning = log_foundation_run(
                    run_type="tabfm",
                    dataset_name=dataset_name,
                    params={**config.model_dump(mode="json"), "dataset_name": dataset_name},
                    metrics=result.metrics,
                    summary_path=result.artifacts["summary"],
                    tracking_uri=settings.tracking.tracking_uri,
                )
                _record_job(store, AppJobType.TABFM, dataset_name, loaded, result, run_id)
                st.session_state["foundation_tabfm_result"] = result
                st.session_state["foundation_tabfm_warning"] = warning
            except Exception as exc:
                log_exception(__import__("logging").getLogger(__name__), exc, operation="foundation.tabfm")
                st.error(safe_error_message(exc))

    result = st.session_state.get("foundation_tabfm_result")
    if result is not None:
        st.success("TabFM evaluation complete.")
        _render_metrics(result.metrics)
        st.dataframe(result.predictions, hide_index=True)
        _download_artifacts(result.artifacts)
        warning = st.session_state.get("foundation_tabfm_warning")
        if warning:
            st.caption(warning)


def _render_timesfm(dataset_name, loaded, settings, store) -> None:  # noqa: ANN001
    dataframe = loaded.dataframe
    st.subheader("TimesFM 2.5")
    st.info("Forecasts and q10–q90 uncertainty are generated locally. Covariates and fine-tuning are not enabled.")
    _dependency_status("timesfm", "uv sync --extra timesfm")
    columns = list(dataframe.columns)
    numeric = list(dataframe.select_dtypes(include="number").columns)
    with st.form("timesfm_run_form"):
        timestamp = st.selectbox("Timestamp column", columns)
        target = st.selectbox("Numeric target", numeric or columns)
        group = st.selectbox("Entity/group column (optional)", [None, *columns])
        left, right = st.columns(2)
        horizon = int(left.number_input("Forecast horizon", min_value=1, max_value=1000, value=12))
        context_length = int(right.number_input("Maximum context", min_value=2, max_value=16384, value=1024))
        frequency = st.text_input("Frequency override (optional)", placeholder="D, W, MS, h…")
        backtest = st.checkbox("Backtest on the final horizon", value=True)
        allow_download = st.checkbox("Allow the first pinned model download (about 1 GB)")
        submitted = st.form_submit_button("Run TimesFM forecast", type="primary")

    if submitted:
        config = TimesFMConfig(
            timestamp_column=timestamp,
            target_column=target,
            group_column=group,
            horizon=horizon,
            context_length=context_length,
            frequency=frequency.strip() or None,
            backtest=backtest,
        )
        with st.spinner("Running pinned TimesFM 2.5 forecast…"):
            try:
                result = _timesfm_service().run(
                    dataframe,
                    config,
                    allow_download=allow_download,
                    output_dir=settings.artifacts.experiments_dir / "foundation",
                )
                run_id, warning = log_foundation_run(
                    run_type="timesfm",
                    dataset_name=dataset_name,
                    params={**config.model_dump(mode="json"), "dataset_name": dataset_name},
                    metrics=result.metrics,
                    summary_path=result.artifacts["summary"],
                    tracking_uri=settings.tracking.tracking_uri,
                )
                _record_job(store, AppJobType.TIMESFM, dataset_name, loaded, result, run_id)
                st.session_state["foundation_timesfm_result"] = result
                st.session_state["foundation_timesfm_config"] = config
                st.session_state["foundation_timesfm_warning"] = warning
            except Exception as exc:
                log_exception(__import__("logging").getLogger(__name__), exc, operation="foundation.timesfm")
                st.error(safe_error_message(exc))

    result = st.session_state.get("foundation_timesfm_result")
    config = st.session_state.get("foundation_timesfm_config")
    if result is not None and config is not None:
        st.success("TimesFM forecast complete.")
        _render_metrics(result.metrics)
        chart_data = result.forecast
        if config.group_column:
            values = list(chart_data[config.group_column].drop_duplicates())
            selected = st.selectbox("Chart entity", values, key="timesfm_chart_group")
            chart_data = chart_data[chart_data[config.group_column] == selected]
        band = (
            alt.Chart(chart_data)
            .mark_area(opacity=0.2)
            .encode(
                x=alt.X(f"{config.timestamp_column}:T", title="Time"),
                y=alt.Y("q10:Q", title=config.target_column),
                y2="q90:Q",
            )
        )
        line = (
            alt.Chart(chart_data)
            .mark_line()
            .encode(
                x=alt.X(f"{config.timestamp_column}:T", title="Time"),
                y=alt.Y("forecast:Q", title=config.target_column),
            )
        )
        st.altair_chart(band + line)
        st.dataframe(result.forecast, hide_index=True)
        _download_artifacts(result.artifacts)
        for warning in [*result.warnings, st.session_state.get("foundation_timesfm_warning")]:
            if warning:
                st.caption(warning)


def _record_job(store, job_type, dataset_name, loaded, result, run_id) -> None:  # noqa: ANN001
    if store is None:
        return
    store.record_job(
        JobRecord(
            job_id=f"{job_type.value}-{uuid4().hex}",
            job_type=job_type,
            status=AppJobStatus.SUCCESS,
            dataset_key=loaded.metadata.content_hash or loaded.metadata.schema_hash,
            dataset_name=dataset_name,
            title=f"{job_type.value.upper()}: {dataset_name}",
            mlflow_run_id=run_id,
            primary_artifact_path=result.artifacts.get("predictions") or result.artifacts.get("forecast"),
            summary_path=result.artifacts.get("summary"),
            metadata={"metrics": result.metrics, "framework": job_type.value},
        )
    )


def _dependency_status(module: str, install_command: str) -> None:
    if importlib.util.find_spec(module) is None:
        st.warning(f"Optional dependency not installed. Run `{install_command}` and restart the app.")
    else:
        st.caption(f":material/check_circle: {module} dependency available · pinned checkpoint cache is reused locally")


def _render_metrics(metrics: dict[str, float | int]) -> None:
    columns = st.columns(min(len(metrics), 4))
    for index, (name, value) in enumerate(metrics.items()):
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        columns[index % len(columns)].metric(name.replace("_", " ").title(), rendered)


def _download_artifacts(artifacts: dict[str, Path]) -> None:
    with st.container(horizontal=True):
        for label, path in artifacts.items():
            if path.is_file():
                st.download_button(
                    f"Download {label}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    key=f"foundation_download_{label}_{path.name}",
                )


__all__ = ["render_foundation_models_page"]
