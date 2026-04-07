"""Streamlit page for model testing – evaluate a trained model on real-world data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from app.prediction import (
    BatchPredictionRequest,
    ModelSourceType,
    PredictionRequest,
    PredictionService,
    PredictionTaskType,
    SchemaValidationMode,
)
from app.security.masking import safe_error_message
from app.pages.dataset_workspace import go_to_page
from app.pages.ui_labels import (
    PREDICTION_TASK_TYPE_LABELS,
    format_enum_value,
    render_decision_support_banner,
    render_model_trust_card,
    SOURCE_TYPE_LABELS,
)
from app.state.session import get_or_init_state
from app.storage import build_metadata_store
from app.tracking.mlflow_query import is_mlflow_available

logger = logging.getLogger(__name__)


def render_model_testing_page(*, _show_header: bool = True) -> None:
    state = get_or_init_state()
    prediction_settings = state.settings.prediction
    tracking_settings = state.settings.tracking
    metadata_store = build_metadata_store(state.settings)
    if _show_header:
        st.title("📊 Test & Evaluate")
        st.caption(
            "Pick a trained model, upload real-world data, "
            "and see how well it performs."
        )

    service = PredictionService(
        artifacts_dir=prediction_settings.artifacts_dir,
        history_path=prediction_settings.history_path,
        schema_validation_mode=SchemaValidationMode(prediction_settings.schema_validation_mode),
        prediction_column_name=prediction_settings.prediction_column_name,
        prediction_score_column_name=prediction_settings.prediction_score_column_name,
        local_model_dirs=prediction_settings.supported_local_model_dirs,
        local_metadata_dirs=prediction_settings.local_model_metadata_dirs,
        tracking_uri=tracking_settings.tracking_uri,
        registry_uri=tracking_settings.registry_uri,
        registry_enabled=tracking_settings.registry_enabled,
        metadata_store=metadata_store,
    )

    # ── Step 1: Model Selection ────────────────────────────────────────
    st.subheader("1. Select a Model")
    model_refs = service.discover_local_models()

    # Also discover benchmark-saved models (joblib .pkl with .json sidecar)
    benchmark_refs = _discover_benchmark_models(state.settings.pycaret.models_dir)

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
    selection_items: list[dict] = []
    for ref in model_refs:
        _task_label = PREDICTION_TASK_TYPE_LABELS.get(ref.task_type.value, format_enum_value(ref.task_type.value))
        selection_items.append({
            "label": f"🔬 {ref.display_name} ({_task_label})",
            "source": "pycaret",
            "ref": ref,
            "benchmark_meta": None,
        })
    for meta in benchmark_refs:
        _raw_task = meta.get('task_type', '?')
        _task_label = PREDICTION_TASK_TYPE_LABELS.get(_raw_task, format_enum_value(_raw_task))
        selection_items.append({
            "label": f"🏁 {meta['model_name']} — {meta.get('dataset_name', '?')} ({_task_label})",
            "source": "benchmark",
            "ref": None,
            "benchmark_meta": meta,
        })

    labels = [item["label"] for item in selection_items]
    selected_label = st.selectbox(
        "Saved model",
        options=labels,
        key="mt_model_select",
        help="🔬 = trained via Train & Tune, 🏁 = trained via Quick Benchmark. Pick the model you want to test.",
    )
    selected_item = selection_items[labels.index(selected_label)]

    # ── Load model ─────────────────────────────────────────────────────
    load_key = f"mt_loaded_{selected_label}"
    if st.session_state.get("mt_loaded_key") != load_key:
        st.session_state.pop("mt_loaded_model", None)
        st.session_state.pop("mt_batch_result", None)
        st.session_state.pop("mt_eval_result", None)

    if st.button("Load Model", key="mt_load"):
        try:
            if selected_item["source"] == "pycaret":
                ref = selected_item["ref"]
                request = PredictionRequest(
                    source_type=ModelSourceType.LOCAL_SAVED_MODEL,
                    model_identifier=ref.load_reference,
                    model_path=ref.model_path,
                    metadata_path=ref.metadata_path,
                    tracking_uri=tracking_settings.tracking_uri,
                    registry_uri=tracking_settings.registry_uri,
                    output_dir=prediction_settings.artifacts_dir,
                    output_stem=prediction_settings.default_output_stem,
                )
                loaded_model = service.load_model(request)
                st.session_state["mt_loaded_model"] = loaded_model
                st.session_state["mt_loaded_key"] = load_key
                st.session_state["mt_loaded_source"] = "pycaret"
            else:
                meta = selected_item["benchmark_meta"]
                loaded = _load_benchmark_model(meta)
                st.session_state["mt_loaded_model"] = loaded
                st.session_state["mt_loaded_key"] = load_key
                st.session_state["mt_loaded_source"] = "benchmark"
            st.success("Model loaded.")
        except Exception as exc:
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
    source_kind = st.session_state.get("mt_loaded_source", "pycaret")

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
        meta = selected_item["benchmark_meta"]
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
                if uploaded.name.endswith((".xlsx", ".xls")):
                    test_df = pd.read_excel(uploaded)
                else:
                    test_df = pd.read_csv(uploaded)
                data_label = uploaded.name
            except Exception as exc:
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
    target_col: str | None = None
    if source_kind == "pycaret" and loaded_model.target_column:
        target_col = loaded_model.target_column
    elif source_kind == "benchmark":
        target_col = selected_item["benchmark_meta"].get("target_column")

    has_ground_truth = target_col is not None and target_col in test_df.columns
    if has_ground_truth:
        st.caption(f"Target column **{target_col}** found in test data — evaluation metrics will be computed.")

    if st.button("Run Test", key="mt_run_test"):
        with st.spinner("Running predictions on test data…"):
            try:
                if source_kind == "pycaret":
                    ref = selected_item["ref"]
                    result = service.predict_batch(
                        BatchPredictionRequest(
                            source_type=ModelSourceType.LOCAL_SAVED_MODEL,
                            model_identifier=ref.load_reference,
                            model_path=ref.model_path,
                            metadata_path=ref.metadata_path,
                            tracking_uri=tracking_settings.tracking_uri,
                            registry_uri=tracking_settings.registry_uri,
                            output_dir=prediction_settings.artifacts_dir,
                            output_stem=prediction_settings.default_output_stem,
                            dataframe=test_df,
                            dataset_name=data_label,
                            input_source_label=data_label,
                        )
                    )
                    st.session_state["mt_batch_result"] = result.scored_dataframe
                    st.session_state["mt_predictions"] = result.scored_dataframe[
                        prediction_settings.prediction_column_name
                    ] if prediction_settings.prediction_column_name in result.scored_dataframe.columns else None
                else:
                    meta = selected_item["benchmark_meta"]
                    native_model = st.session_state["mt_loaded_model"]
                    feature_cols = meta.get("feature_columns", [])
                    available = [c for c in feature_cols if c in test_df.columns]
                    if not available:
                        st.error("Test data does not contain any of the expected input columns.")
                        return
                    X_test = test_df[available]
                    predictions = native_model.predict(X_test)
                    scored = test_df.copy()
                    scored["prediction"] = predictions
                    st.session_state["mt_batch_result"] = scored
                    st.session_state["mt_predictions"] = pd.Series(predictions, name="prediction")

                st.success("Predictions complete.")
            except Exception as exc:
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
        _task = _infer_task_type(source_kind, loaded_model, selected_item)
        try:
            if _task == "classification":
                from sklearn.metrics import accuracy_score as _acc
                _mask = test_df[target_col].notna() & pd.Series(predictions).notna()
                _score_val = _acc(test_df[target_col][_mask], predictions[_mask])
                _score_lbl = f"accuracy of **{_score_val:.2%}**"
                _good = _score_val >= 0.7
            else:
                from sklearn.metrics import r2_score as _r2
                _mask = test_df[target_col].notna() & pd.Series(predictions).notna()
                _score_val = _r2(test_df[target_col][_mask].astype(float), predictions[_mask].astype(float))
                _score_lbl = f"R² of **{_score_val:.4f}**"
                _good = _score_val >= 0.5
            _verdict = (
                "The model performs well on this test set."
                if _good
                else "The model's accuracy is low — consider tuning it or trying a different algorithm."
            )
            st.info(
                f"**What happened:** Generated predictions for **{_n_rows:,}** rows. "
                f"Compared against ground truth and achieved {_score_lbl}.\n\n"
                f"**Looks good?** {_verdict}\n\n"
                "**Next step:** Download the scored file below, or head to **Predict** to run on new data."
            )
        except Exception:
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
        _render_evaluation_metrics(
            y_true=test_df[target_col],
            y_pred=predictions,
            task_type=_infer_task_type(source_kind, loaded_model, selected_item),
        )

    # Download button
    csv_data = scored_df.to_csv(index=False)
    st.download_button(
        "Download scored CSV",
        data=csv_data,
        file_name=f"model_testing_{data_label}.csv",
        mime="text/csv",
        key="mt_download",
    )


# ── Helpers ───────────────────────────────────────────────────────────


def _discover_benchmark_models(models_dir: Path) -> list[dict]:
    """Discover joblib models saved from benchmark with JSON sidecar metadata."""

    results = []
    if not models_dir.exists():
        return results
    for json_path in models_dir.glob("benchmark_*_*.json"):
        pkl_path = json_path.with_suffix(".pkl")
        if pkl_path.exists():
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
                meta["_model_path"] = str(pkl_path)
                meta["_metadata_path"] = str(json_path)
                results.append(meta)
            except Exception:
                continue
    return results


def _load_benchmark_model(meta: dict):
    """Load a benchmark-saved joblib model."""

    import joblib

    return joblib.load(meta["_model_path"])


def _infer_task_type(source_kind: str, loaded_model, selected_item: dict) -> str:
    """Return 'classification' or 'regression' from available info."""

    if source_kind == "pycaret":
        task = loaded_model.task_type.value
        if "classif" in task.lower():
            return "classification"
        if "regress" in task.lower():
            return "regression"
        return task.lower()
    meta = selected_item.get("benchmark_meta", {})
    return meta.get("task_type", "classification").lower()


def _render_evaluation_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    task_type: str,
) -> None:
    """Show evaluation metrics when ground-truth labels are available."""

    st.subheader("Evaluation")
    try:
        if task_type == "classification":
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

            # Drop rows where either is NaN
            mask = y_true.notna() & pd.Series(y_pred).notna()
            yt, yp = y_true[mask], y_pred[mask]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(yt, yp):.4f}")
            col2.metric("Precision", f"{precision_score(yt, yp, average='weighted', zero_division=0):.4f}")
            col3.metric("Recall", f"{recall_score(yt, yp, average='weighted', zero_division=0):.4f}")
            col4.metric("F1", f"{f1_score(yt, yp, average='weighted', zero_division=0):.4f}")
        else:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            mask = y_true.notna() & pd.Series(y_pred).notna()
            yt, yp = y_true[mask].astype(float), y_pred[mask].astype(float)

            col1, col2, col3 = st.columns(3)
            col1.metric("R²", f"{r2_score(yt, yp):.4f}")
            col2.metric("MAE", f"{mean_absolute_error(yt, yp):.4f}")
            col3.metric("RMSE", f"{mean_squared_error(yt, yp, squared=False):.4f}")
    except Exception as exc:
        st.warning(
            f"Could not compute evaluation metrics: {safe_error_message(exc)}\n\n"
            "This can happen when predictions contain unexpected data types. "
            "Your prediction results are still available above."
        )
