"""Unified Predictions page — combines Score New Data and Test & Evaluate."""

from __future__ import annotations

import streamlit as st

from app.pages.workflow_guide import render_workflow_banner


def render_predictions_page() -> None:
    st.title("🔮 Predictions")
    render_workflow_banner(current_step=5)
    st.caption(
        "Choose a saved model, provide data, and get predictions — or test how well a model performs on labelled data."
    )

    predict_tab, evaluate_tab = st.tabs(["🔮 Score New Data", "📊 Test & Evaluate"])

    with predict_tab:
        _render_predict_tab()

    with evaluate_tab:
        _render_evaluate_tab()


def _render_predict_tab() -> None:
    """Delegate to the existing prediction page, skipping the outer title."""
    from app.pages.prediction_page import render_prediction_page as _original

    _original(_show_header=False)


def _render_evaluate_tab() -> None:
    """Delegate to the existing Model Testing page, skipping the outer title."""
    from app.pages.model_testing_page import render_model_testing_page as _original

    _original(_show_header=False)
