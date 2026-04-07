"""Streamlit page for the Train & Tune workflow (PyCaret)."""

from __future__ import annotations

from pathlib import Path

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
from app.path_utils import model_save_name
from app.modeling.pycaret.summary import leaderboard_to_dataframe
from app.pages.dataset_workspace import (
    go_to_page,
    render_dataset_header,
)
from app.pages.workflow_guide import render_next_step_hint, render_workflow_banner
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
    st.title("🧪 Train & Tune")
    render_workflow_banner(current_step=4)
    st.caption(
        "**Build your model** — compare algorithms, fine-tune the winner, and save a production-ready model. "
        "Unlike Quick Benchmark, this step produces a model you can use for predictions."
    )


    if not is_pycaret_available():
        st.error(pycaret_install_guidance())
        return

    # ── Step 1: Choose Your Data ───────────────────────────────────────
    st.subheader("1. Choose Your Data")
    selected_name, loaded_dataset = render_dataset_header("Experiment", key_prefix="experiment", metadata_store=metadata_store)
    if selected_name is None or loaded_dataset is None:
        return

    df = loaded_dataset.dataframe
    metadata = loaded_dataset.metadata
    ensure_dataset_record(metadata_store, loaded_dataset, dataset_name=selected_name)

    st.caption(f"Rows: **{len(df):,}** · Columns: **{len(df.columns):,}**")

    # ── Step 2: Configure ─────────────────────────────────────────────
    st.subheader("2. Configure")
    target_column = st.selectbox(
        "Target column",
        list(df.columns),
        key="exp_target",
        help="The column your model should learn to predict.",
    )
    from app.pages.ui_labels import TASK_TYPE_LABELS, make_format_func
    task_type = ExperimentTaskType(
        st.selectbox(
            "Task type",
            options=[task.value for task in ExperimentTaskType],
            format_func=make_format_func(TASK_TYPE_LABELS),
            index=0,
            key="exp_task_type",
            help="Choose 'Classification' for categories, 'Regression' for numbers, or 'Auto' to let the system decide.",
        )
    )

    # ── Smart defaults (all overridable in Advanced options) ───────────
    default_plots = default_plot_ids_for_task(task_type, settings)
    plot_options = (
        settings.default_plot_ids_classification + settings.default_plot_ids_regression
        if task_type == ExperimentTaskType.AUTO
        else default_plots
    )

    from app.pages.ui_labels import PLOT_LABELS

    # ── Run mode presets ───────────────────────────────────────────────
    from app.pages.glossary import EXPERIMENT_PRESETS

    _exp_preset_names = list(EXPERIMENT_PRESETS.keys())
    exp_run_mode = st.radio(
        "Run mode",
        options=_exp_preset_names,
        index=1,
        horizontal=True,
        key="exp_run_mode",
        help="Quick = fast exploration.  Standard = balanced.  Deep = thorough evaluation with more validation rounds.",
    )
    _exp_preset = EXPERIMENT_PRESETS[exp_run_mode]
    st.caption(_exp_preset["description"])

    with st.expander("Advanced options"):
        # ── Training settings ──────────────────────────────────────────
        st.caption("**Training**")
        col1, col2, col3 = st.columns(3)
        with col1:
            session_id = int(
                st.number_input(
                    "Reproducibility seed",
                    value=int(settings.default_session_id),
                    step=1,
                    key="exp_session_id",
                    help="A number that makes the experiment reproducible. Keep the default unless you need a specific seed.",
                )
            )
        with col2:
            train_size = st.slider(
                "Training data (%)",
                min_value=0.5,
                max_value=0.95,
                value=float(_exp_preset["train_size"]),
                step=0.05,
                key="exp_train_size",
                help="Percentage of data used for training. The rest is held back for validation. 0.7 = 70% training, 30% testing.",
            )
        with col3:
            fold = int(
                st.number_input(
                    "Validation rounds",
                    min_value=2,
                    value=int(_exp_preset["fold"]),
                    step=1,
                    key="exp_fold",
                    help="How many times to re-split and re-test for more reliable scores. More rounds = more reliable but slower. 5 is a good balance.",
                )
            )

        preprocess = st.checkbox(
            "Automatic data preprocessing",
            value=bool(settings.default_preprocess),
            key="exp_preprocess",
            help="Automatically handle missing values, encode categories, and scale numbers before training. Recommended for most datasets.",
        )

        ignore_features_raw = st.text_input(
            "Columns to ignore (comma-separated)",
            key="exp_ignore_features",
            help="List column names that should NOT be used for training — such as ID columns, names, dates, or row numbers. Example: 'id, customer_name, created_at'",
        )

        # ── Scoring ───────────────────────────────────────────────────
        st.caption("**Scoring**")
        score_col1, score_col2, score_col3 = st.columns(3)
        with score_col1:
            compare_metric = st.text_input(
                "Ranking score",
                value=default_compare_metric_for_task(task_type, settings),
                key="exp_compare_metric",
                help="The score used to rank algorithms during the initial comparison (e.g. 'Accuracy' for classification, 'R2' for regression). Leave blank for the default.",
            ).strip()
        with score_col2:
            tune_metric = st.text_input(
                "Tuning score",
                value=default_tune_metric_for_task(task_type, settings),
                key="exp_tune_metric",
                help="The score optimised when fine-tuning the selected model's hyperparameters. Can differ from the ranking score — for example, rank by Accuracy but tune for F1 if you care more about recall.",
            ).strip()
        with score_col3:
            n_select = int(
                st.number_input(
                    "Top models to carry forward",
                    min_value=1,
                    value=int(_exp_preset["n_select"]),
                    step=1,
                    key="exp_n_select",
                    help="How many of the top-ranked algorithms to keep after comparison. These models are available for tuning and saving. More = broader choice, but slower.",
                )
            )

        # ── Performance charts ─────────────────────────────────────────
        st.caption("**Performance charts**")
        selected_plots = st.multiselect(
            "Charts to generate after training",
            options=list(dict.fromkeys(plot_options + ["calibration"])),
            default=default_plots,
            key="exp_plots",
            format_func=make_format_func(PLOT_LABELS),
            help="Visual diagnostics that help you understand model quality: where it's strong, where it struggles, and which features matter most.",
        )

        # ── Hardware ───────────────────────────────────────────────────
        st.caption("**Hardware**")
        from app.gpu import cuda_summary
        gpu_info = cuda_summary()
        from app.pages.ui_labels import GPU_LABELS, make_format_func as _mff
        gpu_options: list[bool | str] = [True, False, "force"]
        default_gpu = settings.default_use_gpu if settings.default_use_gpu in gpu_options else True
        use_gpu = st.selectbox(
            "GPU acceleration",
            options=gpu_options,
            index=gpu_options.index(default_gpu),
            format_func=_mff(GPU_LABELS, fallback_title=False),
            key="exp_use_gpu",
            help=f"CUDA detected: {gpu_info['device_name'] or 'No'}" if gpu_info['cuda_available'] else "No CUDA GPU detected. GPU options will fall back to CPU automatically.",
        )

        # ── Experiment tracking ────────────────────────────────────────
        st.caption("**Experiment tracking**")
        from app.pages.ui_labels import TRACKING_MODE_LABELS
        tracking_mode_value = st.selectbox(
            "Tracking mode",
            options=[mode.value for mode in MLflowTrackingMode],
            format_func=make_format_func(TRACKING_MODE_LABELS, fallback_title=False),
            index=[mode.value for mode in MLflowTrackingMode].index(default_tracking_mode(settings).value),
            key="exp_tracking_mode",
            help="Controls how run details are recorded. 'Automatic' records everything; 'Manual' gives you control; 'Off' skips tracking entirely.",
        )
        tracking_mode = MLflowTrackingMode(tracking_mode_value)
        enable_log_plots = st.checkbox(
            "Save charts to tracking run",
            value=False,
            disabled=tracking_mode != MLflowTrackingMode.PYCARET_NATIVE or not settings.allow_log_plots,
            key="exp_log_plots",
            help="Save performance charts directly to the tracking run. Only available with 'Automatic' tracking.",
        )
        enable_log_profile = st.checkbox(
            "Save data profile to tracking run",
            value=False,
            disabled=tracking_mode != MLflowTrackingMode.PYCARET_NATIVE or not settings.allow_log_profile,
            key="exp_log_profile",
            help="Save a data profile report to the tracking run. Can be large for big datasets.",
        )
        enable_log_data = st.checkbox(
            "Save training data to tracking run",
            value=False,
            disabled=tracking_mode != MLflowTrackingMode.PYCARET_NATIVE or not settings.allow_log_data,
            key="exp_log_data",
            help="Store a copy of the training data inside the tracking run. Useful for reproducibility but increases storage.",
        )

        # ── Auto-save ─────────────────────────────────────────────────
        st.caption("**Auto-save**")
        auto_save_compared_models = st.checkbox(
            "Automatically save all compared models for Predictions",
            value=True,
            key="exp_auto_save_all_models",
            help="When training finishes, save every compared model so you can pick any of them from the Predictions page. Recommended.",
        )
        auto_save_with_snapshots = st.checkbox(
            "Include experiment details when auto-saving",
            value=False,
            key="exp_auto_save_with_snapshots",
            disabled=not auto_save_compared_models,
            help="Save detailed experiment configuration alongside each model. Disabled by default to keep files smaller.",
        )

    # ── Step 3: Start Training ─────────────────────────────────────────────────
    st.subheader("3. Start Training")
    requested_result_key = build_experiment_run_key(selected_name, target_column, task_type)
    bundles = st.session_state.setdefault("experiment_bundles", {})

    if st.button("Start Training", key="exp_run_compare"):
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
        with st.spinner("Training and comparing models — this may take several minutes for large datasets…"):
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
                st.error(
                    f"Experiment stopped: {safe_error_message(exc)}\n\n"
                    "**What to try:** Check your target column and task type, "
                    "or reduce training data % in **Advanced options**."
                )
                return
            except Exception as exc:  # pragma: no cover - UI fallback
                st.error(
                    f"Experiment failed unexpectedly: {safe_error_message(exc)}\n\n"
                    "**What to try:** Verify your data has no formatting issues, "
                    "reduce the number of rows, or reload the dataset from **Load Data**."
                )
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
                    "Training completed, but automatic model save failed: "
                    f"{safe_error_message(exc)}\n\n"
                    "You can still save models manually using the **Save Model** button below."
                )

        bundles[requested_result_key] = bundle
        if saved_count > 0:
            st.success(
                f"Training complete. Saved {saved_count} model(s) for Predictions."
            )
        else:
            st.success("Training complete.")

    bundle = bundles.get(requested_result_key)
    if bundle is None:
        st.info(
            "**Ready to train.**\n\n"
            "Choose a target column above and click **Start Training**.\n\n"
            "**What happens:** The system compares algorithms, fine-tunes the best one, "
            "and saves a production-ready model. This typically takes 2–10 minutes depending on your data size."
        )
        return

    # ── Step 4: Results ───────────────────────────────────────────────
    st.subheader("4. Results")
    _render_bundle(bundle, settings)
    render_next_step_hint(current_step=4)


def _render_bundle(bundle, settings) -> None:  # noqa: ANN001
    from app.pages.ui_labels import format_enum_value, PLOT_LABELS

    summary = bundle.summary

    # ── Plain-English summary ──────────────────────────────────────────
    _best = summary.best_baseline_model_name or "N/A"
    _score = summary.best_baseline_score or "N/A"
    _count = len(bundle.compare_leaderboard)
    _dur = f"{summary.experiment_duration_seconds:.1f}"
    _has_tuned = bundle.tuned_result is not None
    _saved = len(bundle.saved_model_artifacts)

    # Quality assessment
    if _count >= 10:
        _verdict = "Thorough comparison — many algorithms were evaluated. High confidence in the ranking."
    elif _count >= 3:
        _verdict = "Good comparison. For even more confidence, try the **Deep** run mode."
    else:
        _verdict = "Few algorithms were tested. Consider using the **Standard** or **Deep** run mode for broader coverage."

    # Tuning note
    _tuned_note = ""
    if _has_tuned:
        _tuned_note = " A **tuned version** was also produced — check the Tuning Summary below to see if it improved."

    # Saved note
    _saved_note = f" **{_saved}** model(s) were saved and are ready for Predictions." if _saved else ""

    st.info(
        f"**What happened:** Compared **{_count}** algorithms in **{_dur}s**. "
        f"Top performer: **{_best}** (score: **{_score}**).{_tuned_note}{_saved_note}\n\n"
        f"**Quality:** {_verdict}\n\n"
        f"**What to do next:** Use the buttons below to **tune** a model for better results, "
        "**generate charts** to understand its strengths and weaknesses, or **save** it for predictions."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Best baseline", summary.best_baseline_model_name or "N/A", summary.best_baseline_score)
    col2.metric("Compared models", len(bundle.compare_leaderboard))
    col3.metric("Duration (s)", f"{summary.experiment_duration_seconds:.2f}")

    if bundle.mlflow_run_id:
        with st.expander("Technical details", expanded=False):
            st.caption(f"Tracking run ID: `{bundle.mlflow_run_id}` — use this to look up the run in MLflow.")

    with st.expander("📏 Available Scoring Metrics", expanded=False):
        st.caption("These are the metrics the training engine can calculate for your task type.")
        _metrics_df = pd.DataFrame([metric.model_dump(mode="json") for metric in bundle.available_metrics])
        _METRIC_COL_LABELS = {
            "metric_id": "ID",
            "display_name": "Metric",
            "greater_is_better": "Higher = better?",
            "is_custom": "Custom?",
            "raw_values": "Details",
        }
        _metrics_df = _metrics_df.rename(columns=_METRIC_COL_LABELS)
        st.dataframe(
            _metrics_df,
            width="stretch",
        )

    if bundle.compare_leaderboard:
        st.subheader("Leaderboard")
        _exp_lb_df = leaderboard_to_dataframe(bundle.compare_leaderboard)
        st.dataframe(_exp_lb_df, width="stretch")

        # Metric explanations
        from app.pages.glossary import render_metric_legend
        _exp_metric_cols = [c for c in _exp_lb_df.columns if c not in ("Rank", "Model")]
        render_metric_legend(_exp_metric_cols, key_prefix="exp_legend")

        model_options = [row.model_name for row in bundle.compare_leaderboard]
        st.subheader("Actions")
        st.caption("Pick a model from the leaderboard and choose what to do with it.")
        selected_model_name = st.selectbox(
            "Select a model",
            model_options,
            key="exp_selected_model",
            help="Choose an algorithm from the leaderboard to tune, evaluate, or save.",
        )
        selected_row = next(row for row in bundle.compare_leaderboard if row.model_name == selected_model_name)
        selection = ModelSelectionSpec(
            model_id=selected_row.model_id,
            model_name=selected_row.model_name,
            rank=selected_row.rank,
        )

        service = _build_service(get_or_init_state().settings, metadata_store=build_metadata_store(get_or_init_state().settings))
        tune_col, eval_col, save_col, save_all_col = st.columns(4)
        if tune_col.button("🎯 Tune", key="exp_tune_button", help="Fine-tune the selected model's settings for better accuracy. Results appear in the Tuning Summary below."):
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
                st.error(
                    f"Tuning failed: {safe_error_message(exc)}\n\n"
                    "**What to try:** Try a different model, or re-run the experiment with adjusted settings."
                )

        if eval_col.button("📊 Charts", key="exp_eval_button", help="Generate performance charts (ROC curve, confusion matrix, etc.) to visualise model quality."):
            try:
                updated = service.evaluate_model(bundle, selection)
                st.session_state["experiment_bundles"][
                    build_experiment_run_key(
                        bundle.dataset_name or "dataset",
                        bundle.config.target_column,
                        bundle.task_type,
                    )
                ] = updated
                st.success("Performance charts generated.")
                bundle = updated
            except PyCaretExperimentError as exc:
                st.error(
                    f"Plot generation failed: {safe_error_message(exc)}\n\n"
                    "**What to try:** Some plot types may not be available for this model type. Try a different model."
                )

        if save_col.button("💾 Save", key="exp_save_button", type="primary", help="Save this model so you can load it on the Predictions page."):
            try:
                _save_name = model_save_name(bundle.dataset_name, selection.model_name)
                updated = service.finalize_and_save_model(bundle, selection, save_name=_save_name)
                st.session_state["experiment_bundles"][
                    build_experiment_run_key(
                        bundle.dataset_name or "dataset",
                        bundle.config.target_column,
                        bundle.task_type,
                    )
                ] = updated
                st.success("Model saved.")
                bundle = updated
            except PyCaretExperimentError as exc:
                st.error(
                    f"Save failed: {safe_error_message(exc)}\n\n"
                    "**What to try:** Check that the models folder is writable in **Settings**, or try saving a different model."
                )

        if save_all_col.button("💾 Save All", key="exp_save_all_button", help="Save every compared model so you can pick any of them on the Predictions page."):
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
                st.error(
                    f"Batch save failed: {safe_error_message(exc)}\n\n"
                    "**What to try:** Check disk space and folder permissions, or save models individually instead."
                )

    if bundle.tuned_result is not None:
        st.subheader("Tuning Summary")
        st.caption("Comparison of the baseline model vs the fine-tuned version. Look for improved scores in the tuned row.")
        tune_df = pd.DataFrame(
            [
                {"Stage": "Before tuning (baseline)", **bundle.tuned_result.baseline_metrics},
                {"Stage": "After tuning", **bundle.tuned_result.tuned_metrics},
            ]
        )
        st.dataframe(tune_df, width="stretch")
        # Quick interpretation
        _base_vals = list(bundle.tuned_result.baseline_metrics.values())
        _tuned_vals = list(bundle.tuned_result.tuned_metrics.values())
        if _base_vals and _tuned_vals:
            try:
                _first_base = float(_base_vals[0])
                _first_tuned = float(_tuned_vals[0])
                if _first_tuned > _first_base:
                    st.success("Tuning improved the model. Consider saving the tuned version.")
                elif _first_tuned == _first_base:
                    st.info("Tuning produced the same score. The baseline may already be well-optimised.")
                else:
                    st.warning("Tuning did not improve the primary score. The baseline version may be the better choice.")
            except (ValueError, TypeError):
                pass

    if bundle.evaluation_plots:
        st.subheader("Performance Charts")
        st.caption("Visual diagnostics for the selected model — hover over charts for detail.")
        from app.pages.glossary import plot_explanation
        for plot in bundle.evaluation_plots:
            plot_label = PLOT_LABELS.get(plot.plot_id, plot.plot_id.replace("_", " ").title())
            st.markdown(f"**{plot_label}**")
            _plot_desc = plot_explanation(plot.plot_id)
            if _plot_desc:
                st.caption(_plot_desc)
            if plot.path and plot.path.exists():
                st.image(str(plot.path))
            else:
                st.caption("Chart file not found.")

    if bundle.saved_model_metadata is not None:
        st.subheader("Saved Model")
        st.success("Model saved successfully.")
        with st.expander("Saved files", expanded=False):
            st.caption(f"Model file: **{Path(str(bundle.saved_model_metadata.model_path)).name}**")
            if bundle.saved_model_metadata.experiment_snapshot_path is not None:
                st.caption(
                    f"Experiment details: **{Path(str(bundle.saved_model_metadata.experiment_snapshot_path)).name}**"
                )

    if bundle.saved_model_artifacts:
        st.subheader("Models Ready for Predictions")
        st.caption("These models have been saved and can be loaded on the **Predictions** page.")
        saved_rows = []
        for artifact in bundle.saved_model_artifacts:
            saved_rows.append(
                {
                    "Model": artifact.metadata.model_name,
                    "Task": format_enum_value(artifact.metadata.task_type.value),
                    "Target": artifact.metadata.target_column,
                }
            )
        st.dataframe(pd.DataFrame(saved_rows), width="stretch")
        if st.button("Open Predictions", key="exp_open_prediction"):
            go_to_page("Predictions")

    if bundle.artifacts is not None:
        with st.expander("📤 Reports & Downloads", expanded=True):
            for label, path in [
                ("Training setup", bundle.artifacts.setup_json_path),
                ("Metrics CSV", bundle.artifacts.metrics_csv_path),
                ("Leaderboard CSV", bundle.artifacts.compare_csv_path),
                ("Tuning results", bundle.artifacts.tune_json_path),
                ("Summary data", bundle.artifacts.summary_json_path),
                ("Markdown summary", bundle.artifacts.markdown_summary_path),
                ("Model metadata", bundle.artifacts.saved_model_metadata_path),
                ("Experiment details", bundle.artifacts.experiment_snapshot_metadata_path),
            ]:
                if path is not None and path.exists():
                    st.download_button(
                        label=f"Download {label}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        key=f"exp_dl_{label}_{path.name}",
                    )

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