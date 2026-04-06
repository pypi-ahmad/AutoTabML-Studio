"""AutoTabML Studio – Streamlit entry point."""

from __future__ import annotations

import streamlit as st

from app.logging_config import configure_logging
from app.pages.registry import default_page_label, get_page_by_label, get_page_registry, render_registered_page
from app.security.masking import safe_error_message
from app.startup import format_startup_issues, initialize_local_runtime
from app.state.session import get_or_init_state

configure_logging()

st.set_page_config(page_title="AutoTabML Studio", page_icon="🔬", layout="wide")

state = get_or_init_state()
default_nav = default_page_label(state.workspace_mode)
if "nav" not in st.session_state:
    st.session_state["nav"] = default_nav
if "startup_status" not in st.session_state:
    st.session_state["startup_status"] = initialize_local_runtime(
        state.settings,
        include_optional_network_checks=True,
    )

page_registry = get_page_registry()
page_labels = [page.label for page in page_registry]

page = st.sidebar.radio(
    "Navigate",
    options=page_labels,
    key="nav",
)
page_spec = get_page_by_label(page)
st.sidebar.caption(page_spec.description)

startup_status = st.session_state["startup_status"]
if startup_status.issues:
    with st.sidebar.expander("Local Environment", expanded=bool(startup_status.errors)):
        for line in format_startup_issues(startup_status):
            st.caption(line)

try:
    render_registered_page(page)
except Exception as exc:  # pragma: no cover - Streamlit fallback
    st.error(f"Failed to render page '{page}': {safe_error_message(exc)}")
