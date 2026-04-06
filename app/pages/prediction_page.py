"""Streamlit page for local-first prediction and inference workflows."""

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
from app.state.session import get_or_init_state
from app.storage import build_metadata_store
from app.tracking.mlflow_query import is_mlflow_available


def render_prediction_page() -> None:
    state = get_or_init_state()
    prediction_settings = state.settings.prediction
    tracking_settings = state.settings.tracking
    metadata_store = build_metadata_store(state.settings)
    st.title("🔮 Prediction Center")
    st.caption("Load a saved local or MLflow-backed model and score unseen tabular data locally.")

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

    source_options = [ModelSourceType.LOCAL_SAVED_MODEL]
    mlflow_available = is_mlflow_available()
    if mlflow_available:
        source_options.append(ModelSourceType.MLFLOW_RUN_MODEL)
        if tracking_settings.registry_enabled:
            source_options.append(ModelSourceType.MLFLOW_REGISTERED_MODEL)

    source_type = ModelSourceType(
        st.selectbox(
            "Model source",
            options=[option.value for option in source_options],
            format_func=lambda value: value.replace("_", " "),
            key="pred_model_source",
        )
    )

    request_kwargs = _render_model_source_controls(
        service,
        source_type=source_type,
        prediction_settings=prediction_settings,
        tracking_settings=tracking_settings,
        mlflow_available=mlflow_available,
    )

    if request_kwargs is None:
        _clear_loaded_model_if_needed(None)
        st.info("Select a resolvable model source to continue.")
        _render_prediction_history(service)
        return

    request_signature = _request_signature(request_kwargs)
    _clear_loaded_model_if_needed(request_signature)

    if st.button("Load Model", key="pred_load_model"):
        try:
            request = PredictionRequest(**request_kwargs)
            loaded_model = service.load_model(request)
            st.session_state["prediction_loaded_model"] = loaded_model
            st.session_state["prediction_loaded_signature"] = request_signature
            st.session_state.pop("prediction_single_result", None)
            st.session_state.pop("prediction_batch_result", None)
            st.success("Model loaded.")
        except Exception as exc:
            st.error(f"Could not load model: {safe_error_message(exc)}")

    loaded_model = st.session_state.get("prediction_loaded_model")
    if loaded_model is None:
        st.info("Load a model to view metadata and run inference.")
        _render_prediction_history(service)
        return

    _render_loaded_model_metadata(loaded_model)

    prediction_mode = PredictionMode(
        st.radio(
            "Prediction mode",
            options=[PredictionMode.SINGLE_ROW.value, PredictionMode.BATCH.value],
            horizontal=True,
            key="pred_mode",
        )
    )

    if prediction_mode == PredictionMode.SINGLE_ROW:
        _render_single_row_panel(service, request_kwargs, loaded_model)
    else:
        _render_batch_panel(service, request_kwargs, loaded_model)

    _render_prediction_history(service)


def _render_model_source_controls(
    service: PredictionService,
    *,
    source_type: ModelSourceType,
    prediction_settings,
    tracking_settings,
    mlflow_available: bool,
) -> dict | None:  # noqa: ANN001
    base_kwargs = {
        "source_type": source_type,
        "tracking_uri": tracking_settings.tracking_uri,
        "registry_uri": tracking_settings.registry_uri,
        "output_dir": prediction_settings.artifacts_dir,
        "output_stem": prediction_settings.default_output_stem,
    }

    if source_type == ModelSourceType.LOCAL_SAVED_MODEL:
        references = service.discover_local_models()
        selection_mode = st.radio(
            "Local model selection",
            options=["discover", "manual_path"],
            horizontal=True,
            key="pred_local_mode",
        )
        task_type_hint = _prediction_task_type_input("pred_local_task_type_hint")
        if selection_mode == "discover":
            if not references:
                st.info("No saved local models were discovered in the configured directories.")
                return None
            options = {
                f"{item.display_name} | {item.task_type.value} | {item.load_reference}": item
                for item in references
            }
            selected_label = st.selectbox("Discovered local model", options=list(options.keys()), key="pred_local_discovered")
            selected = options[selected_label]
            base_kwargs.update(
                {
                    "model_identifier": selected.load_reference,
                    "model_path": selected.model_path,
                    "metadata_path": selected.metadata_path,
                    "task_type_hint": task_type_hint if selected.task_type == PredictionTaskType.UNKNOWN else None,
                }
            )
            return base_kwargs

        model_path_value = st.text_input("Saved model path", key="pred_local_model_path").strip()
        metadata_path_value = st.text_input("Saved model metadata JSON (optional)", key="pred_local_metadata_path").strip()
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

    if source_type == ModelSourceType.MLFLOW_RUN_MODEL:
        if not mlflow_available:
            st.warning("mlflow is not installed, so MLflow-backed inference is unavailable.")
            return None
        reference_mode = st.radio(
            "MLflow run-model reference",
            options=["model_uri", "run_and_artifact_path"],
            horizontal=True,
            key="pred_mlflow_run_mode",
        )
        task_type_hint = _prediction_task_type_input("pred_mlflow_run_task_type_hint")
        metadata_path_value = st.text_input("Saved model metadata JSON (optional)", key="pred_mlflow_run_metadata").strip()
        if reference_mode == "model_uri":
            model_uri = st.text_input("MLflow model URI", key="pred_mlflow_model_uri").strip()
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

        run_id = st.text_input("MLflow run id", key="pred_mlflow_run_id").strip()
        artifact_path = st.text_input(
            "Artifact path under the run",
            value=prediction_settings.default_mlflow_run_artifact_path,
            key="pred_mlflow_artifact_path",
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

    if source_type == ModelSourceType.MLFLOW_REGISTERED_MODEL:
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
        selected_model_name = st.selectbox("Registered model", options=list(model_lookup.keys()), key="pred_registry_model")
        versions = service.list_registered_model_versions(selected_model_name)
        if not versions:
            st.info("The selected registered model has no versions.")
            return None
        selected_model = model_lookup[selected_model_name]
        alias_options = sorted(selected_model.aliases.keys())
        resolution_mode = st.radio(
            "Resolve registered model by",
            options=["alias", "version"] if alias_options else ["version"],
            horizontal=True,
            key="pred_registry_resolution",
        )
        task_type_hint = _prediction_task_type_input("pred_registry_task_type_hint")
        metadata_path_value = st.text_input("Saved model metadata JSON (optional)", key="pred_registry_metadata").strip()
        base_kwargs.update(
            {
                "registry_model_name": selected_model_name,
                "model_identifier": selected_model_name,
                "metadata_path": Path(metadata_path_value) if metadata_path_value else None,
                "task_type_hint": task_type_hint,
            }
        )
        if resolution_mode == "alias":
            selected_alias = st.selectbox("Alias", options=alias_options, key="pred_registry_alias")
            base_kwargs["registry_alias"] = selected_alias
            return base_kwargs
        selected_version = st.selectbox(
            "Version",
            options=[version.version for version in versions],
            key="pred_registry_version",
        )
        base_kwargs["registry_version"] = selected_version
        return base_kwargs

    return None


def _render_loaded_model_metadata(loaded_model) -> None:  # noqa: ANN001
    st.subheader("Loaded Model")
    col1, col2, col3 = st.columns(3)
    col1.metric("Source", loaded_model.source_type.value)
    col2.metric("Task", loaded_model.task_type.value)
    col3.metric("Feature columns", len(loaded_model.feature_columns))
    st.caption(f"Identifier: `{loaded_model.model_identifier}`")
    st.caption(f"Load reference: `{loaded_model.load_reference}`")
    if loaded_model.target_column:
        st.caption(f"Target column from metadata: `{loaded_model.target_column}`")
    if loaded_model.feature_columns:
        st.markdown("**Expected features**")
        st.code(", ".join(loaded_model.feature_columns), language="text")
    if loaded_model.metadata:
        with st.expander("Model metadata"):
            st.json(loaded_model.metadata)


def _render_single_row_panel(service: PredictionService, request_kwargs: dict, loaded_model) -> None:  # noqa: ANN001
    st.subheader("Single Row Prediction")
    template = {column: None for column in loaded_model.feature_columns} if loaded_model.feature_columns else {}
    row_payload_text = st.text_area(
        "Row JSON",
        value=json.dumps(template, indent=2),
        height=220,
        key="pred_single_row_json",
    )

    if st.button("Run Single Prediction", key="pred_run_single"):
        try:
            row_payload = json.loads(row_payload_text)
            result = service.predict_single(
                SingleRowPredictionRequest(
                    **request_kwargs,
                    row_data=row_payload,
                    input_source_label="manual_row",
                )
            )
            st.session_state["prediction_single_result"] = result
            st.success("Single-row prediction complete.")
        except Exception as exc:
            st.error(f"Single-row prediction failed: {safe_error_message(exc)}")

    result = st.session_state.get("prediction_single_result")
    if result is None:
        return

    result_col1, result_col2, result_col3 = st.columns(3)
    if result.predicted_label is not None:
        result_col1.metric("Predicted label", result.predicted_label)
    else:
        result_col1.metric("Predicted value", result.predicted_value)
    result_col2.metric("Prediction score", f"{result.predicted_score:.4f}" if result.predicted_score is not None else "N/A")
    result_col3.metric("Status", result.summary.status.value)

    st.dataframe(pd.DataFrame([result.scored_row]), width="stretch")
    _render_validation_issues(result.validation)
    _render_summary_block(result.summary)
    _render_artifacts(result.artifacts, key_prefix="pred_single_artifacts")


def _render_batch_panel(service: PredictionService, request_kwargs: dict, loaded_model) -> None:  # noqa: ANN001
    st.subheader("Batch Prediction")
    loaded_datasets = st.session_state.get("loaded_datasets", {})
    batch_source_options = ["upload_file"]
    if loaded_datasets:
        batch_source_options.insert(0, "session_dataset")
    batch_source = st.radio(
        "Batch input source",
        options=batch_source_options,
        horizontal=True,
        key="pred_batch_input_source",
        format_func=lambda value: "session dataset" if value == "session_dataset" else "upload file",
    )

    batch_request_kwargs = {}
    if batch_source == "session_dataset":
        selected_name = st.selectbox("Loaded dataset", options=list(loaded_datasets.keys()), key="pred_session_dataset")
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
                "input_spec": _uploaded_file_to_input_spec(uploaded_file),
                "dataset_name": uploaded_file.name,
                "input_source_label": uploaded_file.name,
            }

    output_path_value = st.text_input("Optional scored output path", key="pred_batch_output_path").strip()

    if st.button("Run Batch Prediction", key="pred_run_batch"):
        if not batch_request_kwargs:
            st.error("Choose a dataset or upload a file before running batch prediction.")
        else:
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
                st.error(f"Batch prediction failed: {safe_error_message(exc)}")

    result = st.session_state.get("prediction_batch_result")
    if result is None:
        return

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Rows scored", result.summary.rows_scored)
    result_col2.metric("Status", result.summary.status.value)
    result_col3.metric("Task", loaded_model.task_type.value)

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
    st.markdown("**Job summary**")
    st.caption(
        f"Mode: {summary.mode.value} | Rows scored: {summary.rows_scored}/{summary.input_row_count} | "
        f"Validation mode: {summary.validation_mode.value}"
    )
    if summary.output_artifact_path is not None:
        st.caption(f"Scored output: `{summary.output_artifact_path}`")
    if summary.warnings:
        for warning in summary.warnings:
            st.warning(warning)


def _render_artifacts(artifacts, *, key_prefix: str) -> None:  # noqa: ANN001
    if artifacts is None:
        return
    with st.expander("Artifacts"):
        artifact_specs = [
            ("Scored CSV", artifacts.scored_csv_path),
            ("Summary JSON", artifacts.summary_json_path),
            ("Metadata JSON", artifacts.metadata_json_path),
            ("Markdown summary", artifacts.markdown_summary_path),
        ]
        for label, path in artifact_specs:
            if path is None:
                continue
            st.markdown(f"{label}: `{path}`")
            if path.exists():
                st.download_button(
                    label=f"Download {label}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    key=f"{key_prefix}_{label}_{path.name}",
                )


def _render_prediction_history(service: PredictionService) -> None:
    st.subheader("Recent Prediction Jobs")
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
        rows.append(
            {
                "Timestamp": entry.timestamp,
                "Status": entry.status.value,
                "Mode": entry.mode.value,
                "Model source": entry.model_source.value,
                "Model": entry.model_identifier,
                "Task": entry.task_type.value,
                "Input source": entry.input_source,
                "Rows": entry.row_count,
                "Output": str(entry.output_artifact_path) if entry.output_artifact_path is not None else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")


def _prediction_task_type_input(key: str):  # noqa: ANN201
    task_type_value = st.selectbox(
        "Task type hint",
        options=[PredictionTaskType.UNKNOWN.value, PredictionTaskType.CLASSIFICATION.value, PredictionTaskType.REGRESSION.value],
        key=key,
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