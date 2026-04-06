"""Streamlit page for the PyCaret experiment lab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.modeling.pycaret.errors import PyCaretExperimentError
from app.modeling.pycaret.schemas import (
    ExperimentCompareConfig,
    ExperimentConfig,
    ExperimentEvaluationConfig,
    ExperimentSetupConfig,
    ExperimentTaskType,
    ExperimentTuneConfig,
    MLflowTrackingMode,
    ModelSelectionSpec,
)
from app.modeling.pycaret.service import PyCaretExperimentService
from app.modeling.pycaret.setup_runner import is_pycaret_available, pycaret_install_guidance
from app.modeling.pycaret.summary import leaderboard_to_dataframe
from app.pages.dataset_workspace import (
    go_to_page,
    render_dataset_header,
)
from app.security.masking import safe_error_message
from app.state.session import get_or_init_state
from app.storage import build_metadata_store, ensure_dataset_record


def build_experiment_run_key(
    dataset_name: str,
    target_column: str,
    requested_task_type: ExperimentTaskType,
) -> str:
    """Return a stable session-state key for experiment results."""

    return f"{dataset_name}::{target_column}::{requested_task_type.value}"


def default_compare_metric_for_task(task_type: ExperimentTaskType, settings) -> str:
    """Return the UI default compare metric for the chosen task."""

    if task_type == ExperimentTaskType.CLASSIFICATION:
        return settings.default_compare_metric_classification
    if task_type == ExperimentTaskType.REGRESSION:
        return settings.default_compare_metric_regression
    return ""


def default_tune_metric_for_task(task_type: ExperimentTaskType, settings) -> str:
    """Return the UI default tune metric for the chosen task."""

    if task_type == ExperimentTaskType.CLASSIFICATION:
        return settings.default_tune_metric_classification
    if task_type == ExperimentTaskType.REGRESSION:
        return settings.default_tune_metric_regression
    return ""


def default_plot_ids_for_task(task_type: ExperimentTaskType, settings) -> list[str]:
    """Return safe default plot ids for the chosen task."""

    if task_type == ExperimentTaskType.CLASSIFICATION:
        return list(settings.default_plot_ids_classification)
    if task_type == ExperimentTaskType.REGRESSION:
        return list(settings.default_plot_ids_regression)
    return []


def default_tracking_mode(settings) -> MLflowTrackingMode:
    """Return the configured default tracking mode with a safe fallback."""

    try:
        return MLflowTrackingMode(settings.default_tracking_mode)
    except ValueError:
        return MLflowTrackingMode.MANUAL


def render_experiment_page() -> None:
    state = get_or_init_state()
    settings = state.settings.pycaret
    metadata_store = build_metadata_store(state.settings)
    st.title("🧪 Experiment Lab")

    if not is_pycaret_available():
        st.error(pycaret_install_guidance())
        return

    selected_name, loaded_dataset = render_dataset_header("Experiment", key_prefix="experiment", metadata_store=metadata_store)
    if selected_name is None or loaded_dataset is None:
        return

    df = loaded_dataset.dataframe
    metadata = loaded_dataset.metadata
    ensure_dataset_record(metadata_store, loaded_dataset, dataset_name=selected_name)

    st.caption(f"Rows: **{len(df):,}** · Columns: **{len(df.columns):,}**")

    target_column = st.selectbox("Target column", list(df.columns), key="exp_target")
    task_type = ExperimentTaskType(
        st.selectbox(
            "Task type",
            options=[task.value for task in ExperimentTaskType],
            index=0,
            key="exp_task_type",
        )
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        session_id = int(
            st.number_input(
                "Session id",
                value=int(settings.default_session_id),
                step=1,
                key="exp_session_id",
            )
        )
    with col2:
        train_size = st.slider(
            "Train size",
            min_value=0.5,
            max_value=0.95,
            value=float(settings.default_train_size),
            step=0.05,
            key="exp_train_size",
        )
    with col3:
        fold = int(
            st.number_input(
                "CV folds",
                min_value=2,
                value=int(settings.default_fold),
                step=1,
                key="exp_fold",
            )
        )

    preprocess = st.checkbox(
        "Enable preprocessing",
        value=bool(settings.default_preprocess),
        key="exp_preprocess",
    )
    compare_metric = st.text_input(
        "Compare metric",
        value=default_compare_metric_for_task(task_type, settings),
        key="exp_compare_metric",
    ).strip()
    tune_metric = st.text_input(
        "Tune metric",
        value=default_tune_metric_for_task(task_type, settings),
        key="exp_tune_metric",
    ).strip()
    n_select = int(
        st.number_input(
            "Top models to keep from compare",
            min_value=1,
            value=3,
            step=1,
            key="exp_n_select",
        )
    )

    default_plots = default_plot_ids_for_task(task_type, settings)
    plot_options = (
        settings.default_plot_ids_classification + settings.default_plot_ids_regression
        if task_type == ExperimentTaskType.AUTO
        else default_plots
    )
    selected_plots = st.multiselect(
        "Evaluation plots",
        options=list(dict.fromkeys(plot_options + ["calibration"])),
        default=default_plots,
        key="exp_plots",
    )

    with st.expander("Advanced options"):
        from app.gpu import cuda_summary
        gpu_info = cuda_summary()
        gpu_options: list[bool | str] = [True, False, "force"]
        gpu_labels = {False: "Off", True: "Auto (use if available)", "force": "Force (fail if unavailable)"}
        default_gpu = settings.default_use_gpu if settings.default_use_gpu in gpu_options else True
        use_gpu = st.selectbox(
            "GPU acceleration",
            options=gpu_options,
            index=gpu_options.index(default_gpu),
            format_func=lambda v: gpu_labels.get(v, str(v)),
            key="exp_use_gpu",
            help=f"CUDA detected: {gpu_info['device_name'] or 'No'}" if gpu_info['cuda_available'] else "No CUDA GPU detected. GPU options will fall back to CPU.",
        )
        ignore_features_raw = st.text_input(
            "Ignore features (comma-separated)",
            key="exp_ignore_features",
        )
        tracking_mode_value = st.selectbox(
            "Tracking mode",
            options=[mode.value for mode in MLflowTrackingMode],
            index=[mode.value for mode in MLflowTrackingMode].index(default_tracking_mode(settings).value),
            key="exp_tracking_mode",
        )
        tracking_mode = MLflowTrackingMode(tracking_mode_value)
        enable_log_plots = st.checkbox(
            "Enable native plot logging",
            value=False,
            disabled=tracking_mode != MLflowTrackingMode.PYCARET_NATIVE or not settings.allow_log_plots,
            key="exp_log_plots",
        )
        enable_log_profile = st.checkbox(
            "Enable native profile logging",
            value=False,
            disabled=tracking_mode != MLflowTrackingMode.PYCARET_NATIVE or not settings.allow_log_profile,
            key="exp_log_profile",
        )
        enable_log_data = st.checkbox(
            "Enable native data logging",
            value=False,
            disabled=tracking_mode != MLflowTrackingMode.PYCARET_NATIVE or not settings.allow_log_data,
            key="exp_log_data",
        )
        auto_save_compared_models = st.checkbox(
            "Auto-save all compared models for Prediction",
            value=True,
            key="exp_auto_save_all_models",
            help=(
                "After compare finishes, finalize and persist every compared model so users can pick them from "
                "Prediction dropdowns."
            ),
        )
        auto_save_with_snapshots = st.checkbox(
            "Include experiment snapshots when auto-saving",
            value=False,
            key="exp_auto_save_with_snapshots",
            disabled=not auto_save_compared_models,
            help="Disabled by default to keep save-all runs faster and smaller on disk.",
        )

    requested_result_key = build_experiment_run_key(selected_name, target_column, task_type)
    bundles = st.session_state.setdefault("experiment_bundles", {})

    if st.button("Run Compare", key="exp_run_compare"):
        service = _build_service(state.settings, metadata_store=metadata_store)
        ignore_features = [item.strip() for item in ignore_features_raw.split(",") if item.strip()]
        config = ExperimentConfig(
            target_column=target_column,
            task_type=task_type,
            mlflow_tracking_mode=tracking_mode,
            setup=ExperimentSetupConfig(
                session_id=session_id,
                train_size=train_size,
                fold=fold,
                fold_strategy=(
                    settings.default_classification_fold_strategy
                    if task_type == ExperimentTaskType.CLASSIFICATION
                    else settings.default_regression_fold_strategy
                    if task_type == ExperimentTaskType.REGRESSION
                    else None
                ),
                ignore_features=ignore_features,
                preprocess=preprocess,
                use_gpu=use_gpu,
                log_experiment=tracking_mode == MLflowTrackingMode.PYCARET_NATIVE,
                log_plots=(selected_plots if enable_log_plots else False),
                log_profile=(enable_log_profile if tracking_mode == MLflowTrackingMode.PYCARET_NATIVE else False),
                log_data=(enable_log_data if tracking_mode == MLflowTrackingMode.PYCARET_NATIVE else False),
            ),
            compare=ExperimentCompareConfig(
                optimize=compare_metric or None,
                n_select=n_select,
            ),
            tune=ExperimentTuneConfig(optimize=tune_metric or None),
            evaluation=ExperimentEvaluationConfig(plots=selected_plots),
        )
        dataset_fingerprint = metadata.content_hash or metadata.schema_hash
        try:
            bundle = service.run_compare_pipeline(
                df,
                config,
                dataset_name=selected_name,
                dataset_fingerprint=dataset_fingerprint,
                execution_backend=state.execution_backend,
                workspace_mode=state.workspace_mode,
            )
        except PyCaretExperimentError as exc:
            st.error(safe_error_message(exc))
            return
        except Exception as exc:  # pragma: no cover - UI fallback
            st.error(f"Experiment run failed unexpectedly: {safe_error_message(exc)}")
            return

        saved_count = 0
        if auto_save_compared_models:
            try:
                bundle = service.finalize_and_save_all_models(
                    bundle,
                    save_name_prefix="auto",
                    include_experiment_snapshots=auto_save_with_snapshots,
                )
                saved_count = len(bundle.saved_model_artifacts)
            except PyCaretExperimentError as exc:
                st.warning(
                    "Compare completed, but automatic model save failed: "
                    f"{safe_error_message(exc)}"
                )

        bundles[requested_result_key] = bundle
        if saved_count > 0:
            st.success(
                f"Experiment compare run complete. Saved {saved_count} model(s) for Prediction Center."
            )
        else:
            st.success("Experiment compare run complete.")

    bundle = bundles.get(requested_result_key)
    if bundle is None:
        st.caption("Run an experiment to inspect metrics, leaderboard, tuning, plots, and saved artifacts.")
        return

    _render_bundle(bundle, settings)


def _render_bundle(bundle, settings) -> None:  # noqa: ANN001
    summary = bundle.summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Best baseline", summary.best_baseline_model_name or "N/A", summary.best_baseline_score)
    col2.metric("Compared models", len(bundle.compare_leaderboard))
    col3.metric("Duration (s)", f"{summary.experiment_duration_seconds:.2f}")

    if bundle.mlflow_run_id:
        st.caption(f"MLflow run id: **{bundle.mlflow_run_id}**")

    st.subheader("Available Metrics")
    st.dataframe(
        pd.DataFrame([metric.model_dump(mode="json") for metric in bundle.available_metrics]),
        width="stretch",
    )

    if bundle.compare_leaderboard:
        st.subheader("Leaderboard")
        st.dataframe(leaderboard_to_dataframe(bundle.compare_leaderboard), width="stretch")

        model_options = [row.model_name for row in bundle.compare_leaderboard]
        selected_model_name = st.selectbox("Model for next step", model_options, key="exp_selected_model")
        selected_row = next(row for row in bundle.compare_leaderboard if row.model_name == selected_model_name)
        selection = ModelSelectionSpec(
            model_id=selected_row.model_id,
            model_name=selected_row.model_name,
            rank=selected_row.rank,
        )

        service = _build_service(get_or_init_state().settings, metadata_store=build_metadata_store(get_or_init_state().settings))
        tune_col, eval_col, save_col, save_all_col = st.columns(4)
        if tune_col.button("Tune Selected", key="exp_tune_button"):
            try:
                updated = service.tune_model(bundle, selection)
                st.session_state["experiment_bundles"][
                    build_experiment_run_key(
                        bundle.dataset_name or "dataset",
                        bundle.config.target_column,
                        bundle.task_type,
                    )
                ] = updated
                st.success("Model tuning complete.")
                bundle = updated
            except PyCaretExperimentError as exc:
                st.error(safe_error_message(exc))

        if eval_col.button("Generate Plots", key="exp_eval_button"):
            try:
                updated = service.evaluate_model(bundle, selection)
                st.session_state["experiment_bundles"][
                    build_experiment_run_key(
                        bundle.dataset_name or "dataset",
                        bundle.config.target_column,
                        bundle.task_type,
                    )
                ] = updated
                st.success("Evaluation plots generated.")
                bundle = updated
            except PyCaretExperimentError as exc:
                st.error(safe_error_message(exc))

        if save_col.button("Finalize And Save", key="exp_save_button"):
            try:
                updated = service.finalize_and_save_model(bundle, selection, save_name="selected_model")
                st.session_state["experiment_bundles"][
                    build_experiment_run_key(
                        bundle.dataset_name or "dataset",
                        bundle.config.target_column,
                        bundle.task_type,
                    )
                ] = updated
                st.success("Model finalized and saved.")
                bundle = updated
            except PyCaretExperimentError as exc:
                st.error(safe_error_message(exc))

        if save_all_col.button("Finalize And Save All", key="exp_save_all_button"):
            try:
                updated = service.finalize_and_save_all_models(bundle, save_name_prefix="manual")
                st.session_state["experiment_bundles"][
                    build_experiment_run_key(
                        bundle.dataset_name or "dataset",
                        bundle.config.target_column,
                        bundle.task_type,
                    )
                ] = updated
                st.success(
                    f"Saved {len(updated.saved_model_artifacts)} model(s)."
                )
                bundle = updated
            except PyCaretExperimentError as exc:
                st.error(safe_error_message(exc))

    if bundle.tuned_result is not None:
        st.subheader("Tuning Summary")
        tune_df = pd.DataFrame(
            [
                {"Stage": "Baseline", **bundle.tuned_result.baseline_metrics},
                {"Stage": "Tuned", **bundle.tuned_result.tuned_metrics},
            ]
        )
        st.dataframe(tune_df, width="stretch")

    if bundle.evaluation_plots:
        st.subheader("Evaluation Plots")
        for plot in bundle.evaluation_plots:
            st.markdown(f"{plot.plot_id}: `{plot.path}`")

    if bundle.saved_model_metadata is not None:
        st.subheader("Saved Model")
        st.markdown(f"Model path: `{bundle.saved_model_metadata.model_path}`")
        if bundle.saved_model_metadata.experiment_snapshot_path is not None:
            st.markdown(
                f"Experiment snapshot: `{bundle.saved_model_metadata.experiment_snapshot_path}` "
                "(original data is not embedded in the snapshot)"
            )

    if bundle.saved_model_artifacts:
        st.subheader("Saved Models For Prediction")
        saved_rows = []
        for artifact in bundle.saved_model_artifacts:
            saved_rows.append(
                {
                    "Model": artifact.metadata.model_name,
                    "Task": artifact.metadata.task_type.value,
                    "Target": artifact.metadata.target_column,
                    "Model path": str(artifact.metadata.model_path),
                    "Metadata": str(artifact.metadata_path) if artifact.metadata_path is not None else "",
                }
            )
        st.dataframe(pd.DataFrame(saved_rows), width="stretch")
        if st.button("Open Prediction Center", key="exp_open_prediction"):
            go_to_page("Prediction")

    if bundle.artifacts is not None:
        with st.expander("Artifacts"):
            for label, path in [
                ("Setup JSON", bundle.artifacts.setup_json_path),
                ("Metrics CSV", bundle.artifacts.metrics_csv_path),
                ("Compare CSV", bundle.artifacts.compare_csv_path),
                ("Tune JSON", bundle.artifacts.tune_json_path),
                ("Summary JSON", bundle.artifacts.summary_json_path),
                ("Markdown Summary", bundle.artifacts.markdown_summary_path),
                ("Saved Model Metadata", bundle.artifacts.saved_model_metadata_path),
                ("Experiment Snapshot Metadata", bundle.artifacts.experiment_snapshot_metadata_path),
            ]:
                if path is not None:
                    st.markdown(f"{label}: `{path}`")

    if bundle.warnings:
        with st.expander("Warnings"):
            for warning in bundle.warnings:
                st.warning(warning)


def _build_service(app_settings, *, metadata_store=None) -> PyCaretExperimentService:  # noqa: ANN001
    settings = app_settings.pycaret
    return PyCaretExperimentService(
        artifacts_dir=settings.artifacts_dir,
        models_dir=settings.models_dir,
        snapshots_dir=settings.snapshots_dir,
        classification_compare_metric=settings.default_compare_metric_classification,
        regression_compare_metric=settings.default_compare_metric_regression,
        classification_tune_metric=settings.default_tune_metric_classification,
        regression_tune_metric=settings.default_tune_metric_regression,
        mlflow_experiment_name=settings.mlflow_experiment_name,
        tracking_uri=app_settings.tracking.tracking_uri,
        registry_uri=app_settings.tracking.registry_uri,
        metadata_store=metadata_store,
    )