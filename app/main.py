"""AutoTabML Studio – Streamlit entry point."""

from __future__ import annotations

import streamlit as st

from app.logging_config import configure_logging
from app.pages.dataset_workspace import render_sidebar_dataset_status
from app.pages.registry import (
    default_page_label,
    get_nav_sections,
    get_page_by_label,
    get_page_registry,
    render_registered_page,
)
from app.security.masking import safe_error_message
from app.startup import format_startup_issues, initialize_local_runtime
from app.state.session import get_or_init_state

configure_logging()

st.set_page_config(page_title="AutoTabML Studio", page_icon="🔬", layout="wide")

state = get_or_init_state()
default_nav = default_page_label(state.workspace_mode)
if "nav" not in st.session_state:
    st.session_state["nav"] = default_nav

# Apply pending navigation before the radio widget renders
if "_pending_nav" in st.session_state:
    st.session_state["nav"] = st.session_state.pop("_pending_nav")

if "startup_status" not in st.session_state:
    st.session_state["startup_status"] = initialize_local_runtime(
        state.settings,
        include_optional_network_checks=True,
    )

page_registry = get_page_registry()
page_labels = [page.label for page in page_registry]

# ── Sectioned sidebar navigation ──────────────────────────────────────
_NAV_ICONS = {
    "Home": "🏠", "Load Data": "📥", "Validation": "✅", "Profiling": "📊",
    "Quick Benchmark": "🏁", "Train & Tune": "🧪",
    "Predictions": "🔮",
    "Models": "📦", "History": "📜", "Compare": "⚖️", "Notebook": "📓",
    "Registry": "🗂️", "Settings": "⚙️",
}
current_nav = st.session_state.get("nav", default_nav)
for section_name, section_pages in get_nav_sections():
    st.sidebar.caption(f"**{section_name}**")
    for spec in section_pages:
        icon = _NAV_ICONS.get(spec.label, "")
        is_active = spec.label == current_nav
        label_text = f"{icon} {spec.label}" if icon else spec.label
        if is_active:
            st.sidebar.markdown(
                f"<div style='padding:4px 8px;background:#e8f0fe;border-radius:6px;"
                f"font-weight:600;margin-bottom:2px'>{label_text}</div>",
                unsafe_allow_html=True,
            )
        else:
            if st.sidebar.button(label_text, key=f"nav_{spec.label}", use_container_width=True):
                st.session_state["nav"] = spec.label
                st.rerun()

page = st.session_state.get("nav", default_nav)
# Validate the nav value still exists in the registry
if page not in page_labels:
    page = default_nav
    st.session_state["nav"] = page
page_spec = get_page_by_label(page)

startup_status = st.session_state["startup_status"]
if startup_status.issues:
    with st.sidebar.expander("Local Environment", expanded=bool(startup_status.errors)):
        for line in format_startup_issues(startup_status):
            st.caption(line)

render_sidebar_dataset_status()
# ── Sidebar glossary ──────────────────────────────────────────────────
from app.pages.glossary import render_glossary_sidebar

render_glossary_sidebar()
# ── Sidebar privacy badge ───────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("🔒 **Private by default** — your data stays on this machine.")

try:
    render_registered_page(page)
except Exception as exc:  # pragma: no cover - Streamlit fallback
    st.error(f"Failed to render page '{page}': {safe_error_message(exc)}")
