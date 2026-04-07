"""Streamlit page for browsing all saved models with details."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.pages.dataset_workspace import go_to_page
from app.pages.ui_labels import PREDICTION_TASK_TYPE_LABELS, format_enum_value, render_model_trust_card
from app.prediction import (
    PredictionService,
    PredictionTaskType,
    SchemaValidationMode,
)
from app.prediction.selectors import discover_local_saved_models
from app.state.session import get_or_init_state
from app.storage import build_metadata_store
from app.tracking.mlflow_query import is_mlflow_available


def render_models_page() -> None:
    state = get_or_init_state()
    prediction_settings = state.settings.prediction
    tracking_settings = state.settings.tracking
    metadata_store = build_metadata_store(state.settings)

    st.title("🗂️ Models")
    st.caption(
        "All the models you’ve trained are listed here. "
        "🔬 = trained via Train & Tune, 🏁 = trained via Quick Benchmark, "
        "📦 = registered in the model registry."
    )

    # ── Discover models from all sources ───────────────────────────────
    pycaret_refs = discover_local_saved_models(
        model_dirs=prediction_settings.supported_local_model_dirs,
        metadata_dirs=prediction_settings.local_model_metadata_dirs,
    )
    benchmark_models = _discover_benchmark_models(state.settings.pycaret.models_dir)

    # DB-tracked models (may overlap with discovered ones)
    # MLflow registry models
    registry_models = []
    if is_mlflow_available() and tracking_settings.registry_enabled:
        try:
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
            registry_models = service.list_registered_models()
        except Exception:
            pass

    total = len(pycaret_refs) + len(benchmark_models) + len(registry_models)
    st.metric("Total Models", total)

    if total == 0:
        st.info(
            "**No saved models yet.**\n\n"
            "Models appear here after you save one from **Quick Benchmark** (Step 3) or **Train & Tune** (Step 4).\n\n"
            "**Recommended:** Load a dataset first, then run a benchmark to find the best algorithm."
        )
        c1, c2, _ = st.columns([2, 2, 4])
        if c1.button("📥 Load Data", key="models_goto_load", type="primary", use_container_width=True):
            go_to_page("Load Data")
        if c2.button("🧪 Train & Tune", key="models_goto_experiment", use_container_width=True):
            go_to_page("Train & Tune")
        return

    # ── Experiment / PyCaret models ────────────────────────────────────
    if pycaret_refs:
        st.subheader(f"🔬 Train & Tune Models ({len(pycaret_refs)})")
        for ref in pycaret_refs:
            _render_pycaret_model_card(ref)

    # ── Benchmark models ───────────────────────────────────────────────
    if benchmark_models:
        st.subheader(f"🏁 Benchmark Models ({len(benchmark_models)})")
        for meta in benchmark_models:
            _render_benchmark_model_card(meta)

    # ── MLflow Registry models ─────────────────────────────────────────
    if registry_models:
        st.subheader(f"📦 Registry Models ({len(registry_models)})")
        for model in registry_models:
            _render_registry_model_card(model)


# ── Card renderers ────────────────────────────────────────────────────


def _render_pycaret_model_card(ref) -> None:  # noqa: ANN001
    """Render an expander card for a PyCaret experiment model."""

    task_label = PREDICTION_TASK_TYPE_LABELS.get(ref.task_type.value, format_enum_value(ref.task_type.value)) if ref.task_type != PredictionTaskType.UNKNOWN else "Unknown"
    target = ref.metadata.get("target_column", "—")
    features = ref.feature_columns
    dataset_name = ref.metadata.get("dataset_name")
    trained_at = ref.metadata.get("trained_at")

    # Build a one-line purpose string
    _purpose_parts = []
    if task_label and task_label != "Unknown":
        _purpose_parts.append(task_label.lower())
    if target and target != "—":
        _purpose_parts.append(f"predicting **{target}**")
    if dataset_name:
        _purpose_parts.append(f"trained on **{dataset_name}**")
    _purpose = ", ".join(_purpose_parts).capitalize() if _purpose_parts else ""

    with st.expander(f"**{ref.display_name}** — {task_label}", expanded=False):
        render_model_trust_card(
            trained_at=trained_at,
            dataset_name=dataset_name,
            task_type=task_label,
            target_column=target,
            feature_count=len(features),
            source_label="Train & Tune",
        )
        if _purpose:
            st.caption(f"🎯 {_purpose}.")

        with st.expander("Saved files", expanded=False):
            st.caption(f"Model file: **{Path(str(ref.model_path)).name}**")
            if ref.metadata_path:
                st.caption(f"Metadata file: **{Path(str(ref.metadata_path)).name}**")
            # Export buttons
            _model_p = Path(str(ref.model_path))
            if _model_p.exists():
                st.download_button(
                    "📤 Download model file",
                    data=_model_p.read_bytes(),
                    file_name=_model_p.name,
                    key=f"mdl_dl_{ref.display_name}_{_model_p.name}",
                )
            _meta_p = Path(str(ref.metadata_path)) if ref.metadata_path else None
            if _meta_p and _meta_p.exists():
                st.download_button(
                    "📤 Download metadata",
                    data=_meta_p.read_bytes(),
                    file_name=_meta_p.name,
                    key=f"mdl_dlm_{ref.display_name}_{_meta_p.name}",
                )

        fingerprint = ref.metadata.get("dataset_fingerprint")
        if fingerprint:
            with st.expander("Training data details", expanded=False):
                st.caption(f"Dataset version: `{fingerprint[:16]}…` — a unique ID that ensures reproducibility.")

        if features:
            with st.expander("Input columns", expanded=False):
                st.code(", ".join(features), language="text")

        dtypes = ref.metadata.get("feature_dtypes", {})
        if dtypes:
            with st.expander("Column types", expanded=False):
                st.dataframe(
                    pd.DataFrame(
                        [{"Feature": k, "Type": v} for k, v in dtypes.items()]
                    ),
                    width="stretch",
                    hide_index=True,
                )


def _render_benchmark_model_card(meta: dict) -> None:
    """Render an expander card for a benchmark-saved model."""

    model_name = meta.get("model_name", "Unknown")
    dataset = meta.get("dataset_name", "—")
    raw_task = meta.get("task_type", "unknown")
    task = PREDICTION_TASK_TYPE_LABELS.get(raw_task, raw_task.title())
    target = meta.get("target_column", "—")
    features = meta.get("feature_columns", [])
    trained_at = meta.get("trained_at")

    # Purpose line
    _purpose_parts = []
    if task and task != "Unknown":
        _purpose_parts.append(task.lower())
    if target and target != "—":
        _purpose_parts.append(f"predicting **{target}**")
    if dataset and dataset != "—":
        _purpose_parts.append(f"trained on **{dataset}**")
    _purpose = ", ".join(_purpose_parts).capitalize() if _purpose_parts else ""

    with st.expander(f"**{model_name}** — {dataset} ({task})", expanded=False):
        render_model_trust_card(
            trained_at=trained_at,
            dataset_name=dataset,
            task_type=task,
            target_column=target,
            feature_count=len(features),
            source_label="Benchmark",
        )
        if _purpose:
            st.caption(f"🎯 {_purpose}.")

        # Provenance cue (factual, not a quality claim)
        if trained_at:
            st.caption(f"📅 Saved from benchmark on {trained_at}. Use **Test & Evaluate** to measure real-world accuracy.")
        else:
            st.caption("⚠️ Training date unknown — consider re-running the benchmark for a fresh model.")

        with st.expander("Saved files", expanded=False):
            _bm_path = Path(str(meta.get("_model_path", "N/A")))
            st.caption(f"Model file: **{_bm_path.name}**")
            if _bm_path.exists():
                st.download_button(
                    "📤 Download model file",
                    data=_bm_path.read_bytes(),
                    file_name=_bm_path.name,
                    key=f"bmdl_dl_{model_name}_{_bm_path.name}",
                )

        split_test = meta.get("split_test_size")
        split_seed = meta.get("split_random_state")
        if split_test is not None:
            st.caption(f"Data split: **{int(split_test * 100)}%** held out for testing (seed {split_seed})")

        fingerprint = meta.get("dataset_fingerprint")
        if fingerprint:
            with st.expander("Training data details", expanded=False):
                st.caption(f"Dataset version: `{fingerprint[:16]}…` — a unique ID that ensures reproducibility.")

        if features:
            with st.expander("Input columns", expanded=False):
                st.code(", ".join(features), language="text")


def _render_registry_model_card(model) -> None:  # noqa: ANN001
    """Render an expander card for an MLflow registry model."""

    aliases = model.aliases or {}
    alias_display = ", ".join(f"`{a}`" for a in sorted(aliases.keys())) if aliases else "none"

    # Determine readiness status from aliases
    _alias_lower = {k.lower() for k in aliases}
    if "champion" in _alias_lower:
        _status_badge = "⭐ Champion (production-ready)"
    elif "candidate" in _alias_lower:
        _status_badge = "🧪 Candidate (testing)"
    else:
        _status_badge = "📦 Registered"

    with st.expander(f"**{model.name}** — {_status_badge}", expanded=False):
        col1, col2 = st.columns(2)
        col1.markdown(f"**Status:** {_status_badge}")
        col1.markdown(f"**Version labels:** {alias_display}")
        if model.description:
            col2.markdown(f"**Description:** {model.description}")
        else:
            col2.caption("No description — add one when registering or promoting.")
        if hasattr(model, "tags") and model.tags:
            st.caption(f"Tags: {model.tags}")


# ── Helpers ───────────────────────────────────────────────────────────


def _discover_benchmark_models(models_dir: Path) -> list[dict]:
    """Discover joblib models saved from benchmark with JSON sidecar metadata."""

    results = []
    if not models_dir.exists():
        return results
    for json_path in models_dir.glob("*.json"):
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("source") != "benchmark":
            continue
        pkl_path = json_path.with_suffix(".pkl")
        if pkl_path.exists():
            meta["_model_path"] = str(pkl_path)
            meta["_metadata_path"] = str(json_path)
            results.append(meta)
    return results
