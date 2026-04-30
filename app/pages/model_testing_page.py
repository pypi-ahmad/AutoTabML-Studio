"""Streamlit page for model testing – evaluate a trained model on real-world data."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from app.modeling.benchmark.persistence import discover_saved_benchmark_models
from app.pages.dataset_workspace import go_to_page
from app.pages.ui_cache import get_prediction_service, get_prediction_workflow_service
from app.pages.ui_errors import log_ui_debug_exception, log_ui_exception
from app.pages.ui_labels import (
    SOURCE_TYPE_LABELS,
    render_decision_support_banner,
    render_model_trust_card,
)
from app.security.masking import safe_error_message
from app.security.safe_csv import dataframe_to_safe_csv
from app.state.session import get_or_init_state

logger = logging.getLogger(__name__)


def render_model_testing_page(*, _show_header: bool = True) -> None:
    state = get_or_init_state()
    workflow = get_prediction_workflow_service()
    execution_config = workflow.build_execution_config(state.settings.prediction, state.settings.tracking)
    if _show_header:
        st.title("📊 Test & Evaluate")
        st.caption(
            "Pick a trained model, upload real-world data, "
            "and see how well it performs."
        )

    service = get_prediction_service(state.settings)

    # ── Step 1: Model Selection ────────────────────────────────────────
    st.subheader("1. Select a Model")
    model_refs = service.discover_local_models()

    # Also discover benchmark-saved models (joblib .pkl with .json sidecar)
    benchmark_refs = discover_saved_benchmark_models(state.settings.pycaret.models_dir)

    if not model_refs and not benchmark_refs:
        st.info(
            "**No saved models found.**\n\n"
            "Train and save a model first (via **Train & Tune** or **Quick Benchmark**), "
            "then come back here to test it against real-world data."
        )
        if st.button("🧪 Train & Tune", key="mt_goto_experiment", type="primary"):
            go_to_page("Train & Tune")
        return

    # Build unified selection list
    selection_items = workflow.build_model_testing_selections(model_refs, benchmark_refs)
    labels = [item.label for item in selection_items]
    selected_label = st.selectbox(
        "Saved model",
        options=labels,
        key="mt_model_select",
        help="🔬 = trained via Train & Tune, 🏁 = trained via Quick Benchmark. Pick the model you want to test.",
    )
    selected_item = next(item for item in selection_items if item.label == selected_label)

    # ── Load model ─────────────────────────────────────────────────────
    load_key = workflow.build_model_testing_load_key(selected_item)
    workflow.reset_model_testing_state(st.session_state, load_key)

    if st.button("Load Model", key="mt_load"):
        try:
            loaded_model = workflow.load_model_testing_model(
                selection=selected_item,
                prediction_service=service,
                config=execution_config,
                trusted_model_roots=[state.settings.pycaret.models_dir],
            )
            st.session_state["mt_loaded_model"] = loaded_model
            st.session_state["mt_loaded_key"] = load_key
            st.session_state["mt_loaded_source"] = selected_item.source_kind
            st.success("Model loaded.")
        except Exception as exc:
            log_ui_exception(exc, operation="model_testing.load_model", context={"source": selected_item.source_kind})
            st.error(
                f"**Could not load model:** {safe_error_message(exc)}\n\n"
                "**Likely cause:** The model file may be missing or corrupted.\n\n"
                "**What to try:** Re-train from **Train & Tune** or re-run a **Quick Benchmark**."
            )

    loaded_model = st.session_state.get("mt_loaded_model")
    if loaded_model is None:
        st.info("Select and load a model to continue.")
        return

    # ── Model info ─────────────────────────────────────────────────────
    source_kind = st.session_state.get("mt_loaded_source", selected_item.source_kind)

    if source_kind == "pycaret":
        _meta = loaded_model.metadata if hasattr(loaded_model, "metadata") and isinstance(loaded_model.metadata, dict) else {}
        render_model_trust_card(
            trained_at=_meta.get("trained_at"),
            dataset_name=_meta.get("dataset_name"),
            task_type=loaded_model.task_type.value if hasattr(loaded_model, "task_type") else None,
            target_column=loaded_model.target_column if hasattr(loaded_model, "target_column") else None,
            feature_count=len(loaded_model.feature_columns) if hasattr(loaded_model, "feature_columns") else None,
            source_label=SOURCE_TYPE_LABELS.get(getattr(loaded_model, "source_type", None) and loaded_model.source_type.value, "Local"),
        )
        if loaded_model.feature_columns:
            with st.expander("Expected features"):
                st.code(", ".join(loaded_model.feature_columns), language="text")
    else:
        meta = selected_item.benchmark_metadata or {}
        render_model_trust_card(
            trained_at=meta.get("trained_at"),
            dataset_name=meta.get("dataset_name"),
            task_type=meta.get("task_type"),
            target_column=meta.get("target_column"),
            feature_count=len(meta.get("feature_columns", [])),
            source_label="Benchmark",
        )
        if meta.get("feature_columns"):
            with st.expander("Expected features"):
                st.code(", ".join(meta["feature_columns"]), language="text")

    render_decision_support_banner()

    # ── Step 2: Test Data ──────────────────────────────────────────────
    st.subheader("2. Upload Test Data")
    data_source = st.radio(
        "Data source",
        options=["Upload file", "Loaded dataset"],
        horizontal=True,
        key="mt_data_source",
        help="Choose whether to upload a new file or reuse a dataset already loaded in this session.",
    )

    test_df: pd.DataFrame | None = None
    data_label = "uploaded"

    if data_source == "Upload file":
        uploaded = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "txt", "tsv", "xlsx", "xls"],
            key="mt_upload",
        )
        if uploaded is not None:
            try:
                test_df = workflow.load_uploaded_test_dataframe(uploaded)
                data_label = uploaded.name
            except Exception as exc:
                log_ui_exception(exc, operation="model_testing.parse_uploaded_file", context={"filename": uploaded.name})
                st.error(f"Failed to parse file: {safe_error_message(exc)}")
    else:
        loaded_datasets = st.session_state.get("loaded_datasets", {})
        if not loaded_datasets:
            st.info("No datasets loaded yet.")
            if st.button("Go to Load Data", key="mt_goto_load_data"):
                go_to_page("Load Data")
        else:
            selected_ds = st.selectbox(
                "Loaded dataset", options=list(loaded_datasets.keys()), key="mt_session_ds",
                help="Choose one of the datasets you loaded earlier.",
            )
            test_df = loaded_datasets[selected_ds].dataframe
            data_label = selected_ds

    if test_df is None:
        st.info("Upload a file or select a loaded dataset above to provide test data.")
        return

    st.caption(f"Test data: **{len(test_df):,}** rows × **{len(test_df.columns):,}** columns")
    st.dataframe(test_df.head(10), width="stretch")

    # ── Step 3: Run Test ───────────────────────────────────────────────
    st.subheader("3. Run Predictions")

    # Detect if target column is present (for evaluation)
    target_col = workflow.target_column_for_testing(source_kind, loaded_model, selected_item)
    has_ground_truth = target_col is not None and target_col in test_df.columns
    if has_ground_truth:
        st.caption(f"Target column **{target_col}** found in test data — evaluation metrics will be computed.")

    if st.button("Run Test", key="mt_run_test"):
        with st.spinner("Running predictions on test data…"):
            try:
                run_result = workflow.run_model_testing_predictions(
                    selection=selected_item,
                    loaded_model=loaded_model,
                    test_dataframe=test_df,
                    data_label=data_label,
                    prediction_service=service,
                    config=execution_config,
                )
                st.session_state["mt_batch_result"] = run_result.scored_dataframe
                st.session_state["mt_predictions"] = run_result.predictions

                st.success("Predictions complete.")
            except Exception as exc:
                log_ui_exception(exc, operation="model_testing.run_predictions", context={"source": source_kind})
                st.error(
                    f"**Prediction failed:** {safe_error_message(exc)}\n\n"
                    "**Likely cause:** Your test data columns may not match what the model expects.\n\n"
                    "**What to try:** View the expected features in the model card above "
                    "and check that your file has matching column names."
                )

    scored_df = st.session_state.get("mt_batch_result")
    predictions = st.session_state.get("mt_predictions")
    if scored_df is None:
        return

    # ── Step 4: Results ────────────────────────────────────────────────
    st.subheader("4. Results")

    _n_rows = len(scored_df)
    if has_ground_truth and predictions is not None:
        try:
            evaluation = workflow.evaluate_predictions(
                y_true=test_df[target_col],
                y_pred=predictions,
                task_type=workflow.infer_model_testing_task_type(source_kind, loaded_model, selected_item),
            )
            primary_value = evaluation.metrics[evaluation.primary_metric_key]
            if evaluation.primary_metric_key == "accuracy":
                score_label = f"accuracy of **{primary_value:.2%}**"
            else:
                score_label = f"R² of **{primary_value:.4f}**"
            st.info(
                f"**What happened:** Generated predictions for **{_n_rows:,}** rows. "
                f"Compared against ground truth and achieved {score_label}.\n\n"
                f"**Looks good?** {evaluation.verdict}\n\n"
                "**Next step:** Download the scored file below, or head to **Predict** to run on new data."
            )
        except Exception as exc:
            log_ui_debug_exception(
                exc,
                operation="model_testing.preview_metrics",
                context={"task_type": workflow.infer_model_testing_task_type(source_kind, loaded_model, selected_item)},
            )
            st.info(
                f"**What happened:** Generated predictions for **{_n_rows:,}** rows with ground truth available.\n\n"
                "**Next step:** Review the metrics below, then download the scored file."
            )
    else:
        st.info(
            f"**What happened:** Generated predictions for **{_n_rows:,}** rows (no ground-truth column to compare against).\n\n"
            "**Next step:** Download the scored CSV below to review the predictions."
        )

    st.dataframe(scored_df.head(50), width="stretch")

    if has_ground_truth and predictions is not None:
        try:
            evaluation = workflow.evaluate_predictions(
                y_true=test_df[target_col],
                y_pred=predictions,
                task_type=workflow.infer_model_testing_task_type(source_kind, loaded_model, selected_item),
            )
            _render_evaluation_metrics(evaluation)
        except Exception as exc:
            log_ui_exception(
                exc,
                operation="model_testing.render_evaluation_metrics",
                context={"task_type": workflow.infer_model_testing_task_type(source_kind, loaded_model, selected_item)},
            )
            st.warning(
                f"Could not compute evaluation metrics: {safe_error_message(exc)}\n\n"
                "This can happen when predictions contain unexpected data types. "
                "Your prediction results are still available above."
            )

    # Download button
    csv_data = dataframe_to_safe_csv(scored_df, index=False)
    st.download_button(
        "Download scored CSV",
        data=csv_data,
        file_name=f"model_testing_{data_label}.csv",
        mime="text/csv",
        key="mt_download",
    )


def _render_evaluation_metrics(evaluation) -> None:  # noqa: ANN001
    """Show evaluation metrics when ground-truth labels are available."""

    st.subheader("Evaluation")
    if evaluation.task_type == "classification":
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{evaluation.metrics['accuracy']:.4f}")
        col2.metric("Precision", f"{evaluation.metrics['precision']:.4f}")
        col3.metric("Recall", f"{evaluation.metrics['recall']:.4f}")
        col4.metric("F1", f"{evaluation.metrics['f1']:.4f}")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{evaluation.metrics['r2']:.4f}")
    col2.metric("MAE", f"{evaluation.metrics['mae']:.4f}")
    col3.metric("RMSE", f"{evaluation.metrics['rmse']:.4f}")
