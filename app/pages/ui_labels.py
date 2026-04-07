"""Shared UI label helpers for converting technical metadata into business-friendly displays."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

# ── Technical key → business label ────────────────────────────────────

_META_KEY_LABELS: dict[str, str] = {
    # Dataset metadata
    "source_type": "Source",
    "source_locator": "File / URL",
    "content_hash": "Data fingerprint",
    "schema_hash": "Structure fingerprint",
    "dataset_fingerprint": "Dataset version ID",
    "dataset_name": "Training dataset",
    "row_count": "Rows",
    "column_count": "Columns",
    "normalization_actions": "Cleanup steps",
    "loaded_at": "Loaded at",
    "source_details": "Source details",
    "columns": "Columns",
    "column_types": "Column types",
    # Model / experiment metadata
    "model_name": "Model",
    "model_count": "Models tested",
    "best_model_name": "Best model",
    "best_score": "Best score",
    "ranking_metric": "Ranking score",
    "ranking_direction": "Sort direction",
    "task_type": "Task",
    "target_column": "Target column",
    "trained_at": "Trained",
    "feature_columns": "Input columns",
    "feature_dtypes": "Column types",
    "train_row_count": "Training rows",
    "test_row_count": "Test rows",
    "benchmark_duration_seconds": "Duration (seconds)",
    "experiment_duration_seconds": "Duration (seconds)",
    "fastest_model_name": "Fastest model",
    "fastest_model_time_seconds": "Fastest time (seconds)",
    "split_test_size": "Test split",
    "split_random_state": "Random seed",
    # MLflow / tracking
    "mlflow_run_id": "Tracking run ID",
    "mlflow_experiment_id": "Tracking experiment ID",
    # Validation
    "passed_count": "Checks passed",
    "warning_count": "Warnings",
    "failed_count": "Checks failed",
    "failed_row_count": "Failed rows",
    "missing_value_pct": "Missing values (%)",
    "total_checks": "Total checks",
}

# Keys that are internal / noisy and should be hidden from the summary view
_HIDDEN_KEYS: set[str] = {
    "source_locator",
    "content_hash",
    "schema_hash",
    "normalization_actions",
    "source_details",
    "columns",
    "column_types",
    "feature_columns",
    "feature_dtypes",
    "_model_path",
    "_metadata_path",
}


# ── Shared enum / option label maps ──────────────────────────────────
# Centralised so pages don't duplicate these dicts.

PROVIDER_LABELS: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "ollama": "Ollama (local)",
}

BACKEND_LABELS: dict[str, str] = {
    "colab_mcp": "Cloud (Google Colab)",
    "local": "Local (this machine)",
}

MODE_LABELS: dict[str, str] = {
    "dashboard": "Dashboard",
    "notebook": "Notebook",
}

TASK_TYPE_LABELS: dict[str, str] = {
    "classification": "Classification — predict a category (e.g. Yes/No)",
    "regression": "Regression — predict a number (e.g. price)",
    "auto": "Auto-detect — let the system decide based on your data",
    "unknown": "Auto-detect",
}

PREDICTION_TASK_TYPE_LABELS: dict[str, str] = {
    "unknown": "Auto-detect",
    "classification": "Classification",
    "regression": "Regression",
}

SOURCE_TYPE_LABELS: dict[str, str] = {
    "LOCAL_SAVED_MODEL": "Saved model (local)",
    "MLFLOW_RUN_MODEL": "Experiment run (MLflow)",
    "MLFLOW_REGISTERED_MODEL": "Registered model (MLflow)",
}

DATASET_SOURCE_LABELS: dict[str, str] = {
    "csv": "CSV file",
    "excel": "Excel file",
    "url": "URL",
    "uci": "UCI Repository",
    "upload": "Upload",
}

PLOT_LABELS: dict[str, str] = {
    "auc": "ROC Curve",
    "confusion_matrix": "Confusion Matrix",
    "threshold": "Threshold Plot",
    "pr": "Precision-Recall Curve",
    "error": "Prediction Error",
    "class_report": "Classification Report",
    "rfe": "Feature Importance",
    "learning": "Learning Curve",
    "manifold": "Manifold Plot",
    "calibration": "Calibration Plot",
    "vc": "Validation Curve",
    "dimension": "Dimension Plot",
    "feature": "Feature Importance",
    "feature_all": "All Feature Importance",
    "parameter": "Model Parameters",
    "lift": "Lift Chart",
    "gain": "Gain Chart",
    "tree": "Decision Tree",
    "ks": "KS Statistic",
    "residuals": "Residual Plot",
    "cooks": "Cook's Distance",
}

TRACKING_MODE_LABELS: dict[str, str] = {
    "pycaret_native": "Automatic tracking",
    "manual": "Manual (app controls tracking)",
    "disabled": "Off (no experiment tracking)",
}

GPU_LABELS: dict[str | bool, str] = {
    False: "Off",
    True: "Auto (use if available)",
    "force": "Force (fail if unavailable)",
}

PROMOTION_LABELS: dict[str, str] = {
    "PROMOTE_TO_CHAMPION": "⭐ Promote to Champion (production-ready)",
    "PROMOTE_TO_CANDIDATE": "🧪 Promote to Candidate (testing)",
    "ARCHIVE": "📦 Archive (remove from active use)",
    "DEMOTE_CHAMPION": "⬇️ Remove Champion status",
    "DEMOTE_CANDIDATE": "⬇️ Remove Candidate status",
}


# ── Shared formatting utilities ──────────────────────────────────────


def format_enum_value(value: str) -> str:
    """Turn a snake_case enum value into a readable title.

    >>> format_enum_value("benchmark_completed")
    'Benchmark Completed'
    """
    return value.replace("_", " ").strip().title()


def make_format_func(labels: dict, *, fallback_title: bool = True):
    """Create a ``format_func`` callback for Streamlit widgets.

    Parameters
    ----------
    labels:
        Mapping from raw option values to display strings.
    fallback_title:
        If *True*, unknown values are title-cased via :func:`format_enum_value`.
        If *False*, unknown values are returned as-is.

    Returns
    -------
    A callable suitable for ``st.selectbox(format_func=…)`` etc.
    """
    if fallback_title:
        return lambda v: labels.get(v, format_enum_value(str(v)))
    return lambda v: labels.get(v, v)


def _label_for_key(key: str) -> str:
    """Return a human-friendly label for a metadata key."""
    if key in _META_KEY_LABELS:
        return _META_KEY_LABELS[key]
    # Auto-humanize: replace underscores and title-case
    return key.replace("_", " ").strip().capitalize()


def _format_value(key: str, value: Any) -> str:
    """Format a metadata value for display."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if "pct" in key or "percent" in key or "size" in key:
            return f"{value:.1%}" if value <= 1.0 else f"{value:.2f}"
        return f"{value:.4f}" if abs(value) < 100 else f"{value:,.2f}"
    if isinstance(value, list):
        if len(value) <= 5:
            return ", ".join(str(v) for v in value)
        return f"{len(value)} items"
    if isinstance(value, dict):
        return f"{len(value)} entries"
    text = str(value)
    # Truncate very long strings (hashes, paths)
    if len(text) > 80:
        return text[:40] + "…" + text[-12:]
    return text


def render_metadata_table(
    metadata: dict[str, Any],
    *,
    title: str = "Details",
    show_hidden: bool = False,
) -> None:
    """Render metadata as a clean two-column table with business labels.

    Hidden/internal keys are excluded from the summary view and placed
    in a nested "Full details" expander.
    """
    visible: list[tuple[str, str]] = []
    hidden: list[tuple[str, str]] = []

    for key, value in metadata.items():
        label = _label_for_key(key)
        formatted = _format_value(key, value)
        if key in _HIDDEN_KEYS and not show_hidden:
            hidden.append((label, formatted))
        else:
            visible.append((label, formatted))

    if visible:
        for label, value in visible:
            st.markdown(f"**{label}:** {value}")

    if hidden:
        with st.expander("Full details"):
            for label, value in hidden:
                st.markdown(f"**{label}:** {value}")


def render_metadata_expander(
    metadata: dict[str, Any],
    *,
    expander_label: str = "Details",
    expanded: bool = False,
) -> None:
    """Render a metadata dict inside an expander with business-friendly labels."""
    with st.expander(expander_label, expanded=expanded):
        render_metadata_table(metadata)


# ── Trust cue helpers ─────────────────────────────────────────────────


def _format_trained_at(trained_at: str | None) -> str:
    """Format an ISO timestamp into a friendly relative label."""
    if not trained_at:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(trained_at)
        now = datetime.now(dt.tzinfo)
        delta = now - dt
        if delta.days == 0:
            return "Today"
        if delta.days == 1:
            return "Yesterday"
        if delta.days < 30:
            return f"{delta.days} days ago"
        if delta.days < 365:
            months = delta.days // 30
            return f"{months} month{'s' if months != 1 else ''} ago"
        return dt.strftime("%b %d, %Y")
    except Exception:
        return trained_at[:10] if len(trained_at) >= 10 else trained_at


def _model_status_label(trained_at: str | None) -> tuple[str, str]:
    """Return (badge, description) based on model freshness.

    All locally-trained models are experimental / decision-support only.
    """
    return "🧪 Experimental", "Not independently validated — use as decision support only."


def render_model_trust_card(
    *,
    trained_at: str | None = None,
    dataset_name: str | None = None,
    task_type: str | None = None,
    target_column: str | None = None,
    feature_count: int | None = None,
    fingerprint: str | None = None,
    source_label: str | None = None,
) -> None:
    """Render a compact trust-cue card for a model.

    Shows: purpose, training dataset, date, approval status, and
    a decision-support disclaimer.
    """
    badge, disclaimer = _model_status_label(trained_at)

    # Row 1: key facts
    cols = st.columns(4)
    cols[0].markdown(f"**Status:** {badge}")
    cols[1].markdown(f"**Trained:** {_format_trained_at(trained_at)}")
    cols[2].markdown(f"**Dataset:** {dataset_name or '—'}")
    if source_label:
        cols[3].markdown(f"**Source:** {source_label}")
    elif feature_count is not None:
        cols[3].markdown(f"**Features:** {feature_count}")

    # Row 2: purpose line
    purpose_parts: list[str] = []
    if task_type:
        purpose_parts.append(task_type.replace("_", " ").title())
    if target_column:
        purpose_parts.append(f"predicting **{target_column}**")
    if purpose_parts:
        st.caption(f"Purpose: {' — '.join(purpose_parts)}")

    # Row 3: disclaimer
    st.caption(f"⚠️ {disclaimer}")


def render_decision_support_banner() -> None:
    """Render a one-line reminder that predictions are decision support, not decisions."""
    st.caption(
        "ℹ️ Model predictions are **decision support** — always review results "
        "before acting on them."
    )
