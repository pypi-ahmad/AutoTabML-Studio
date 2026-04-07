"""Streamlit page for local-first prediction workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.prediction import (
    BatchPredictionRequest,
    ModelSourceType,
    PredictionMode,
    PredictionRequest,
    PredictionService,
    PredictionTaskType,
    SchemaValidationMode,
    SingleRowPredictionRequest,
)
from app.security.masking import safe_error_message
from app.pages.workflow_guide import render_workflow_banner
from app.pages.dataset_workspace import go_to_page, uploaded_file_to_input_spec
from app.state.session import get_or_init_state
from app.storage import build_metadata_store
from app.tracking.mlflow_query import is_mlflow_available


from app.pages.ui_labels import (
    PREDICTION_TASK_TYPE_LABELS,
    SOURCE_TYPE_LABELS,
    format_enum_value,
    make_format_func,
)


def render_prediction_page(*, _show_header: bool = True) -> None:
    state = get_or_init_state()
    prediction_settings = state.settings.prediction
    tracking_settings = state.settings.tracking
    metadata_store = build_metadata_store(state.settings)
    if _show_header:
        st.title("🔮 Predictions")
        render_workflow_banner(current_step=5)
        st.caption(
            "Choose a saved model, provide data, and get predictions."
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

    # ── Step 1: Pick a saved model ─────────────────────────────────────
    st.subheader("1. Pick a model")

    request_kwargs = _render_primary_model_picker(
        service, prediction_settings=prediction_settings, tracking_settings=tracking_settings
    )

    # ── Advanced model sources (manual path, MLflow) ───────────────────
    adv_kwargs = _render_advanced_model_sources(
        service, prediction_settings=prediction_settings, tracking_settings=tracking_settings
    )
    if adv_kwargs is not None:
        request_kwargs = adv_kwargs

    if request_kwargs is None:
        _clear_loaded_model_if_needed(None)
        _render_prediction_history(service)
        return

    request_signature = _request_signature(request_kwargs)
    _clear_loaded_model_if_needed(request_signature)

    if st.button("Load Model", key="pred_load_model", type="primary"):
        try:
            request = PredictionRequest(**request_kwargs)
            loaded_model = service.load_model(request)
            st.session_state["prediction_loaded_model"] = loaded_model
            st.session_state["prediction_loaded_signature"] = request_signature
            st.session_state.pop("prediction_single_result", None)
            st.session_state.pop("prediction_batch_result", None)
            st.success("Model loaded.")
        except Exception as exc:
            st.error(
                f"**Could not load model:** {safe_error_message(exc)}\n\n"
                "**Likely cause:** The model file may be missing, moved, or incompatible with this environment.\n\n"
                "**What to try:** Re-save the model from **Train & Tune**, or check the file path in **Advanced model sources**."
            )

    loaded_model = st.session_state.get("prediction_loaded_model")
    if loaded_model is None:
        st.info(
            "Select a model above and click **Load Model** to start making predictions.\n\n"
            "Once loaded, you can score a whole file or predict one record at a time."
        )
        _render_prediction_history(service)
        return

    _render_loaded_model_metadata(loaded_model)

    # ── Step 2: Provide data & predict ─────────────────────────────────
    st.subheader("2. Provide data & predict")

    batch_tab, single_tab = st.tabs(["📄 Score a file", "✏️ Predict one record"])

    with batch_tab:
        _render_batch_panel(service, request_kwargs, loaded_model)

    with single_tab:
        _render_single_row_panel(service, request_kwargs, loaded_model)

    _render_prediction_history(service)


# ── Model picker: primary (saved models) + advanced expander ───────────


def _render_primary_model_picker(
    service: PredictionService,
    *,
    prediction_settings,  # noqa: ANN001
    tracking_settings,  # noqa: ANN001
) -> dict | None:
    """Render the default saved-model browser.

    Advanced sources (manual path, MLflow) are available inside a collapsed
    expander so the primary business flow stays simple.
    """
    base_kwargs = {
        "source_type": ModelSourceType.LOCAL_SAVED_MODEL,
        "tracking_uri": tracking_settings.tracking_uri,
        "registry_uri": tracking_settings.registry_uri,
        "output_dir": prediction_settings.artifacts_dir,
        "output_stem": prediction_settings.default_output_stem,
    }

    references = service.discover_local_models()
    if not references:
        st.info(
            "**No saved models found yet.**\n\n"
            "To make predictions you need a trained model. "
            "Go to **Train & Tune** (Step 4) to train and save one, "
            "or run a **Quick Benchmark** (Step 3) and save the best result.\n\n"
            "If you haven't loaded data yet, start with **Load Data** (Step 1)."
        )
        c1, c2, _ = st.columns([2, 2, 4])
        if c1.button("🧪 Train & Tune", key="pred_goto_experiment", type="primary", use_container_width=True):
            go_to_page("Train & Tune")
        if c2.button("📥 Load Data", key="pred_goto_load", use_container_width=True):
            go_to_page("Load Data")
        return base_kwargs

    options = {
        f"{item.display_name} ({PREDICTION_TASK_TYPE_LABELS.get(item.task_type.value, format_enum_value(item.task_type.value))})": item
        for item in references
    }
    selected_label = st.selectbox(
        "Saved model",
        options=list(options.keys()),
        key="pred_local_discovered",
        help="These are models previously saved from Quick Benchmark or Train & Tune.",
    )
    selected = options[selected_label]
    task_type_hint = None
    if selected.task_type == PredictionTaskType.UNKNOWN:
        task_type_hint = _prediction_task_type_input("pred_local_task_type_hint")
    base_kwargs.update(
        {
            "model_identifier": selected.load_reference,
            "model_path": selected.model_path,
            "metadata_path": selected.metadata_path,
            "task_type_hint": task_type_hint,
        }
    )
    return base_kwargs


def _render_advanced_model_sources(
    service: PredictionService,
    *,
    prediction_settings,  # noqa: ANN001
    tracking_settings,  # noqa: ANN001
) -> dict | None:
    """Render manual-path, MLflow run, and MLflow registry options inside a
    collapsed expander so they don't clutter the primary flow."""

    mlflow_available = is_mlflow_available()

    with st.expander("Advanced model sources", expanded=False):
        source_options = ["manual_path"]
        _ADV_LABELS = {
            "manual_path": "Enter a file path",
            "MLFLOW_RUN_MODEL": "MLflow experiment run",
            "MLFLOW_REGISTERED_MODEL": "MLflow registered model",
        }
        if mlflow_available:
            source_options.append("MLFLOW_RUN_MODEL")
            if tracking_settings.registry_enabled:
                source_options.append("MLFLOW_REGISTERED_MODEL")

        adv_choice = st.radio(
            "Model source",
            options=source_options,
            format_func=lambda v: _ADV_LABELS.get(v, v),
            key="pred_adv_source",
            horizontal=True,
        )

        base_kwargs: dict = {
            "tracking_uri": tracking_settings.tracking_uri,
            "registry_uri": tracking_settings.registry_uri,
            "output_dir": prediction_settings.artifacts_dir,
            "output_stem": prediction_settings.default_output_stem,
        }

        if adv_choice == "manual_path":
            base_kwargs["source_type"] = ModelSourceType.LOCAL_SAVED_MODEL
            task_type_hint = _prediction_task_type_input("pred_manual_task_type_hint")
            model_path_value = st.text_input("Saved model path", key="pred_local_model_path", help="Full path to your saved model file on disk.").strip()
            metadata_path_value = st.text_input("Model metadata file (optional)", key="pred_local_metadata_path", help="Path to a metadata file with model details (features, target column, etc.). Leave blank if unavailable.").strip()
            if not model_path_value:
                return None
            base_kwargs.update(
                {
                    "model_identifier": model_path_value,
                    "model_path": Path(model_path_value),
                    "metadata_path": Path(metadata_path_value) if metadata_path_value else None,
                    "task_type_hint": task_type_hint,
                }
            )
            return base_kwargs

        if adv_choice == "MLFLOW_RUN_MODEL":
            base_kwargs["source_type"] = ModelSourceType.MLFLOW_RUN_MODEL
            if not mlflow_available:
                st.warning("Experiment tracking is not set up, so MLflow-based predictions are unavailable. Ask your administrator to configure it.")
                return None
            reference_mode = st.radio(
                "How to find the model",
                options=["model_uri", "run_and_artifact_path"],
                horizontal=True,
                key="pred_mlflow_run_mode",
                format_func=lambda v: "Paste a model URI" if v == "model_uri" else "Specify run + folder",
            )
            task_type_hint = _prediction_task_type_input("pred_mlflow_run_task_type_hint")
            metadata_path_value = st.text_input("Model metadata file (optional)", key="pred_mlflow_run_metadata", help="Path to a metadata file with model details. Leave blank if unavailable.").strip()
            if reference_mode == "model_uri":
                model_uri = st.text_input("Model URI", key="pred_mlflow_model_uri", help="A model URI like 'runs:/<run_id>/model' or 'models:/<name>/<version>'.").strip()
                if not model_uri:
                    return None
                base_kwargs.update(
                    {
                        "model_uri": model_uri,
                        "model_identifier": model_uri,
                        "metadata_path": Path(metadata_path_value) if metadata_path_value else None,
                        "task_type_hint": task_type_hint,
                    }
                )
                return base_kwargs

            run_id = st.text_input("Run ID", key="pred_mlflow_run_id", help="The ID of the tracking run that produced the model.").strip()
            artifact_path = st.text_input(
                "Model folder name",
                value=prediction_settings.default_mlflow_run_artifact_path,
                key="pred_mlflow_artifact_path",
                help="Folder name inside the run where the model is stored (usually 'model').",
            ).strip()
            if not run_id or not artifact_path:
                return None
            base_kwargs.update(
                {
                    "run_id": run_id,
                    "artifact_path": artifact_path,
                    "model_identifier": run_id,
                    "metadata_path": Path(metadata_path_value) if metadata_path_value else None,
                    "task_type_hint": task_type_hint,
                }
            )
            return base_kwargs

        if adv_choice == "MLFLOW_REGISTERED_MODEL":
            base_kwargs["source_type"] = ModelSourceType.MLFLOW_REGISTERED_MODEL
            if not tracking_settings.registry_enabled:
                st.warning("Model registry is disabled in settings.")
                return None
            try:
                models = service.list_registered_models()
            except Exception as exc:
                st.warning(f"Model registry is unavailable for the current MLflow configuration: {safe_error_message(exc)}")
                return None
            if not models:
                st.info("No registered models are available yet.")
                return None

            model_lookup = {model.name: model for model in models}
            selected_model_name = st.selectbox("Registered model", options=list(model_lookup.keys()), key="pred_registry_model", help="Pick a model from the registry to use for predictions.")
            versions = service.list_registered_model_versions(selected_model_name)
            if not versions:
                st.info("The selected registered model has no versions.")
                return None
            selected_model = model_lookup[selected_model_name]
            alias_options = sorted(selected_model.aliases.keys())
            resolution_mode = st.radio(
                "Look up model by",
                options=["alias", "version"] if alias_options else ["version"],
                horizontal=True,
                key="pred_registry_resolution",
                format_func=lambda v: "Version label (e.g. 'production')" if v == "alias" else "Version number",
            )
            task_type_hint = _prediction_task_type_input("pred_registry_task_type_hint")
            metadata_path_value = st.text_input("Model metadata file (optional)", key="pred_registry_metadata", help="Path to a metadata file with model details. Leave blank if unavailable.").strip()
            base_kwargs.update(
                {
                    "registry_model_name": selected_model_name,
                    "model_identifier": selected_model_name,
                    "metadata_path": Path(metadata_path_value) if metadata_path_value else None,
                    "task_type_hint": task_type_hint,
                }
            )
            if resolution_mode == "alias":
                selected_alias = st.selectbox("Version label", options=alias_options, key="pred_registry_alias", help="Named tag like 'production' or 'staging' pointing to a specific version.")
                base_kwargs["registry_alias"] = selected_alias
                return base_kwargs
            selected_version = st.selectbox(
                "Version",
                options=[version.version for version in versions],
                key="pred_registry_version",
                help="The version number of the model to use.",
            )
            base_kwargs["registry_version"] = selected_version
            return base_kwargs

    return None


def _render_loaded_model_metadata(loaded_model) -> None:  # noqa: ANN001
    from app.pages.ui_labels import render_decision_support_banner, render_model_trust_card

    st.subheader("Loaded Model")
    _meta = loaded_model.metadata if isinstance(loaded_model.metadata, dict) else {}
    render_model_trust_card(
        trained_at=_meta.get("trained_at"),
        dataset_name=_meta.get("dataset_name"),
        task_type=loaded_model.task_type.value,
        target_column=loaded_model.target_column,
        feature_count=len(loaded_model.feature_columns),
        source_label=SOURCE_TYPE_LABELS.get(loaded_model.source_type.value, format_enum_value(loaded_model.source_type.value)),
    )
    render_decision_support_banner()
    if loaded_model.target_column:
        st.caption(f"Target column: **{loaded_model.target_column}**")
    if loaded_model.feature_columns:
        with st.expander("Expected features", expanded=False):
            st.code(", ".join(loaded_model.feature_columns), language="text")
    with st.expander("Model details", expanded=False):
        if loaded_model.metadata:
            from app.pages.ui_labels import render_metadata_table
            _meta = loaded_model.metadata if isinstance(loaded_model.metadata, dict) else {"value": loaded_model.metadata}
            render_metadata_table(_meta)


def _render_single_row_panel(service: PredictionService, request_kwargs: dict, loaded_model) -> None:  # noqa: ANN001
    st.caption("Fill in values for a single record and get an instant prediction.")

    feature_columns = loaded_model.feature_columns or []
    feature_dtypes: dict[str, str] = {}
    if loaded_model.metadata and isinstance(loaded_model.metadata, dict):
        feature_dtypes = loaded_model.metadata.get("feature_dtypes", {})

    # Decide whether we can show a per-column form
    use_form = bool(feature_columns)

    if use_form:
        row_payload = _render_column_form(feature_columns, feature_dtypes)
        with st.expander("Edit as JSON (advanced)", expanded=False):
            st.caption("Edit the JSON directly if you prefer, or copy it for use elsewhere.")
            row_json_text = st.text_area(
                "Prediction input data",
                value=json.dumps(row_payload, indent=2, default=str),
                height=180,
                key="pred_single_row_json",
                help="Auto-generated from the form above. You can edit it manually.",
            )
    else:
        st.info("The model has no saved column information. Enter values as JSON below.")
        row_json_text = st.text_area(
            "Prediction input data",
            value="{}",
            height=220,
            key="pred_single_row_json",
            help="Enter a JSON object with one key-value pair per input column, e.g. {\"age\": 30, \"income\": 50000}.",
        )
        row_payload = None

    if st.button("Run Prediction", key="pred_run_single", type="primary"):
        try:
            if use_form:
                # Always use the form values — the JSON text area is for
                # viewing/copying only (its value goes stale when Streamlit
                # rerenders because the widget keeps its session-state value).
                final_payload = row_payload
            else:
                final_payload = json.loads(row_json_text)

            result = service.predict_single(
                SingleRowPredictionRequest(
                    **request_kwargs,
                    row_data=final_payload,
                    input_source_label="manual_row",
                )
            )
            st.session_state["prediction_single_result"] = result
            st.success("Prediction complete.")
        except json.JSONDecodeError:
            st.error(
                "**Invalid JSON format.**\n\n"
                "**Likely cause:** The JSON text has a syntax error (missing quote, comma, or bracket).\n\n"
                "**What to try:** Use the column form above instead, or fix the JSON syntax."
            )
        except Exception as exc:
            st.error(
                f"**Prediction failed:** {safe_error_message(exc)}\n\n"
                "**Likely cause:** The input data may not match what the model expects.\n\n"
                "**What to try:** Check that you've filled in values for all required columns. "
                "See the **Expected features** section in the model card above."
            )

    result = st.session_state.get("prediction_single_result")
    if result is None:
        return

    # Plain-English result
    _pred = result.predicted_label if result.predicted_label is not None else result.predicted_value
    _score = f"{result.predicted_score:.4f}" if result.predicted_score is not None else None
    _summary_parts = [f"**Prediction:** {_pred}"]
    if _score:
        _summary_parts.append(f"**Confidence score:** {_score}")
    st.success(" · ".join(_summary_parts))

    result_col1, result_col2, result_col3 = st.columns(3)
    if result.predicted_label is not None:
        result_col1.metric("Predicted label", result.predicted_label)
    else:
        result_col1.metric("Predicted value", result.predicted_value)
    result_col2.metric("Confidence score", _score or "N/A")
    result_col3.metric("Status", format_enum_value(result.summary.status.value))

    st.dataframe(pd.DataFrame([result.scored_row]), width="stretch")
    _render_validation_issues(result.validation)
    _render_summary_block(result.summary)
    _render_artifacts(result.artifacts, key_prefix="pred_single_artifacts")


def _render_column_form(feature_columns: list[str], feature_dtypes: dict[str, str]) -> dict:
    """Render per-column inputs and return the row payload dict."""
    st.caption("Fill in one value per column. Leave blank for missing values.")
    row_payload: dict = {}
    # Render in a 2-column grid for compactness
    col_pairs = [feature_columns[i : i + 2] for i in range(0, len(feature_columns), 2)]
    for pair in col_pairs:
        cols = st.columns(len(pair))
        for col_widget, col_name in zip(cols, pair):
            dtype_str = feature_dtypes.get(col_name, "").lower()
            with col_widget:
                if any(t in dtype_str for t in ("int", "float", "double", "numeric")):
                    val = st.text_input(
                        col_name,
                        value="",
                        key=f"pred_col_{col_name}",
                        help=f"Numeric column ({dtype_str}). Enter a number.",
                    )
                    if val.strip():
                        try:
                            row_payload[col_name] = float(val) if "float" in dtype_str or "double" in dtype_str else int(val)
                        except ValueError:
                            row_payload[col_name] = val.strip()
                    else:
                        row_payload[col_name] = None
                else:
                    val = st.text_input(
                        col_name,
                        value="",
                        key=f"pred_col_{col_name}",
                        help=f"Text/category column ({dtype_str or 'text'}).",
                    )
                    row_payload[col_name] = val.strip() if val.strip() else None
    return row_payload


def _render_batch_panel(service: PredictionService, request_kwargs: dict, loaded_model) -> None:  # noqa: ANN001
    st.caption("Upload a CSV / Excel file or pick a loaded dataset to score in bulk.")
    loaded_datasets = st.session_state.get("loaded_datasets", {})
    batch_source_options = ["upload_file"]
    if loaded_datasets:
        batch_source_options.append("session_dataset")
    _BATCH_SOURCE_LABELS = {
        "upload_file": "Upload a CSV or Excel file",
        "session_dataset": "Use a loaded dataset",
    }
    batch_source = st.radio(
        "Where is your data?",
        options=batch_source_options,
        horizontal=True,
        key="pred_batch_input_source",
        format_func=lambda value: _BATCH_SOURCE_LABELS.get(value, value),
        help="Upload new data or pick a dataset you already loaded earlier in this session.",
    )

    batch_request_kwargs = {}
    if batch_source == "session_dataset":
        selected_name = st.selectbox("Loaded dataset", options=list(loaded_datasets.keys()), key="pred_session_dataset", help="Choose one of the datasets you loaded earlier.")
        selected_dataset = loaded_datasets[selected_name]
        batch_request_kwargs = {
            "dataframe": selected_dataset.dataframe,
            "dataset_name": selected_name,
            "input_source_label": selected_name,
        }
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "txt", "tsv", "xlsx", "xls"],
            key="pred_batch_upload",
        )
        if uploaded_file is not None:
            batch_request_kwargs = {
                "input_spec": uploaded_file_to_input_spec(uploaded_file),
                "dataset_name": uploaded_file.name,
                "input_source_label": uploaded_file.name,
            }

    with st.expander("Output options", expanded=False):
        output_path_value = st.text_input("Custom output file path (optional)", key="pred_batch_output_path", help="Where to save the scored output file. Leave blank to use the default location.").strip()

    if st.button("Run Predictions", key="pred_run_batch", type="primary"):
        if not batch_request_kwargs:
            st.error("Choose a dataset or upload a file before running predictions.")
        else:
            with st.spinner("Running predictions…"):
                try:
                    result = service.predict_batch(
                        BatchPredictionRequest(
                            **request_kwargs,
                            **batch_request_kwargs,
                            output_path=Path(output_path_value) if output_path_value else None,
                        )
                    )
                    st.session_state["prediction_batch_result"] = result
                    st.success("Batch prediction complete.")
                except Exception as exc:
                    st.error(
                        f"**Prediction failed:** {safe_error_message(exc)}\n\n"
                        "**Likely cause:** Your data columns may not match what the model expects, "
                        "or the file format may be unsupported.\n\n"
                        "**What to try:** Check the **Expected features** in the model card, "
                        "or try re-loading the model."
                    )

    result = st.session_state.get("prediction_batch_result")
    if result is None:
        return

    # Plain-English summary
    _n = result.summary.rows_scored
    _status = format_enum_value(result.summary.status.value)
    _task = PREDICTION_TASK_TYPE_LABELS.get(loaded_model.task_type.value, format_enum_value(loaded_model.task_type.value))
    st.info(
        f"**What happened:** Scored **{_n:,}** rows using a **{_task}** model.\n\n"
        "**Next step:** Download the scored file below, or review the predictions in the table."
    )

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Rows scored", f"{_n:,}")
    result_col2.metric("Status", _status)
    result_col3.metric("Task", _task)

    st.dataframe(result.scored_dataframe.head(25), width="stretch")
    _render_validation_issues(result.validation)
    _render_summary_block(result.summary)
    _render_artifacts(result.artifacts, key_prefix="pred_batch_artifacts")


def _render_validation_issues(validation) -> None:  # noqa: ANN001
    if not validation.issues:
        return
    with st.expander("Validation"):
        for issue in validation.issues:
            if issue.severity.value == "error":
                st.error(issue.message)
            elif issue.severity.value == "warning":
                st.warning(issue.message)
            else:
                st.info(issue.message)


def _render_summary_block(summary) -> None:  # noqa: ANN001
    _VALIDATION_MODE_LABELS = {
        "strict": "Strict",
        "warn": "Warn only",
        "disabled": "Off",
    }
    with st.expander("Job summary", expanded=False):
        st.caption(
            f"Mode: {format_enum_value(summary.mode.value)} | "
            f"Rows scored: {summary.rows_scored}/{summary.input_row_count} | "
            f"Validation: {_VALIDATION_MODE_LABELS.get(summary.validation_mode.value, format_enum_value(summary.validation_mode.value))}"
        )
        if summary.output_artifact_path is not None:
            st.caption(f"Output file: `{Path(summary.output_artifact_path).name}`")
        if summary.warnings:
            for warning in summary.warnings:
                st.warning(warning)


def _render_artifacts(artifacts, *, key_prefix: str) -> None:  # noqa: ANN001
    if artifacts is None:
        return
    with st.expander("Reports & Downloads"):
        artifact_specs = [
            ("Scored CSV", artifacts.scored_csv_path),
            ("Summary data", artifacts.summary_json_path),
            ("Run metadata", artifacts.metadata_json_path),
            ("Markdown summary", artifacts.markdown_summary_path),
        ]
        for label, path in artifact_specs:
            if path is None:
                continue
            if path.exists():
                st.download_button(
                    label=f"Download {label}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    key=f"{key_prefix}_{label}_{path.name}",
                )


def _render_prediction_history(service: PredictionService) -> None:
    with st.expander("📋 Recent Prediction Jobs", expanded=False):
        try:
            entries = service.list_history(limit=15)
        except Exception as exc:
            st.warning(f"Prediction history could not be loaded: {safe_error_message(exc)}")
            return
        if not entries:
            st.caption("No prediction jobs have been recorded yet.")
            return

        rows = []
        for entry in entries:
            _output_name = Path(str(entry.output_artifact_path)).name if entry.output_artifact_path else ""
            rows.append(
                {
                    "Timestamp": entry.timestamp,
                    "Status": format_enum_value(entry.status.value),
                    "Mode": format_enum_value(entry.mode.value),
                    "Model source": SOURCE_TYPE_LABELS.get(entry.model_source.value, format_enum_value(entry.model_source.value)),
                    "Model": entry.model_identifier,
                    "Task": PREDICTION_TASK_TYPE_LABELS.get(entry.task_type.value, format_enum_value(entry.task_type.value)),
                    "Input": entry.input_source,
                    "Rows": entry.row_count,
                    "Output": _output_name,
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch")


_TASK_TYPE_LABELS: dict[str, str] = PREDICTION_TASK_TYPE_LABELS


def _prediction_task_type_input(key: str):  # noqa: ANN201
    task_type_value = st.selectbox(
        "Task type hint",
        options=[PredictionTaskType.UNKNOWN.value, PredictionTaskType.CLASSIFICATION.value, PredictionTaskType.REGRESSION.value],
        key=key,
        format_func=make_format_func(PREDICTION_TASK_TYPE_LABELS),
        help="Helps the system load the model correctly. Use 'Auto-detect' if unsure.",
    )
    if task_type_value == PredictionTaskType.UNKNOWN.value:
        return None
    return PredictionTaskType(task_type_value)


def _request_signature(request_kwargs: dict) -> str:
    normalized = json.dumps(request_kwargs, default=str, sort_keys=True)
    return normalized


def _clear_loaded_model_if_needed(current_signature: str | None) -> None:
    previous_signature = st.session_state.get("prediction_loaded_signature")
    if previous_signature == current_signature:
        return
    st.session_state.pop("prediction_loaded_model", None)
    st.session_state.pop("prediction_single_result", None)
    st.session_state.pop("prediction_batch_result", None)
    if current_signature is None:
        st.session_state.pop("prediction_loaded_signature", None)