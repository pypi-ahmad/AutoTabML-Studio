"""AutoTabML Studio – Streamlit entry point."""

from __future__ import annotations

import logging

import streamlit as st

from app.errors import log_exception
from app.logging_config import configure_logging
from app.pages.dataset_workspace import render_sidebar_dataset_status
from app.pages.registry import (
    build_streamlit_navigation,
)
from app.security.masking import safe_error_message
from app.startup import format_startup_issues, initialize_local_runtime
from app.state.session import get_or_init_state

configure_logging()

logger = logging.getLogger(__name__)

st.set_page_config(page_title="AutoTabML Studio", page_icon="🔬", layout="wide")

state = get_or_init_state()

# Streamlit reruns this entire script on every user interaction, so the guard
# below prevents initialize_local_runtime from running more than once per
# browser session.  The result (startup issues / warnings) is retained across
# reruns through session_state.
if "startup_status" not in st.session_state:
    st.session_state["startup_status"] = initialize_local_runtime(
        state.settings,
        include_optional_network_checks=True,
    )

navigation_sections, page_objects = build_streamlit_navigation(state.workspace_mode)
st.session_state["_page_objects"] = page_objects
# Programmatic navigation: st.switch_page cannot be called before st.navigation,
# so callers set `st.session_state["_pending_nav"] = page_key` and trigger a
# rerun.  On the next rerun we pop the key and switch before navigation renders.
pending_page = st.session_state.pop("_pending_nav", None)
if pending_page in page_objects:
    st.switch_page(page_objects[pending_page])
page = st.navigation(navigation_sections, position="top")

startup_status = st.session_state["startup_status"]
if startup_status.issues:
    with st.sidebar.expander("Local Environment", expanded=bool(startup_status.errors)):
        for line in format_startup_issues(startup_status):
            st.caption(line)

render_sidebar_dataset_status()
# Imported late so that st.set_page_config (called at module scope above) has
# already executed before glossary's own Streamlit calls run.
# ── Sidebar glossary ──────────────────────────────────────────────────
from app.pages.glossary import render_glossary_sidebar

render_glossary_sidebar()
# ── Sidebar privacy badge ───────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("🔒 **Private by default** — your data stays on this machine.")

try:
    page.run()
except Exception as exc:  # pragma: no cover - Streamlit fallback
    log_exception(
        logger,
        exc,
        operation="streamlit.render_page",
        context={"page": page.title},
    )
    st.error(f"Failed to render page '{page.title}': {safe_error_message(exc)}")
