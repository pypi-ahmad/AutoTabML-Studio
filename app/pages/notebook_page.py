"""Notebook mode page – Colab MCP default, local fallback.

When the execution backend is **colab_mcp** (the default), this page manages
a connection to Google Colab via the Model Context Protocol and lets the user
execute code cells remotely.  When **local** is selected, the page presents a
lightweight local-notebook placeholder.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging

import streamlit as st

from app.backends import build_backend
from app.config.enums import ExecutionBackend
from app.state.session import get_or_init_state

logger = logging.getLogger(__name__)

_BACKEND_KEY = "notebook_backend"
_SESSION_KEY = "notebook_session_info"


def _run_async(coro):  # noqa: ANN001
    """Execute *coro* from synchronous Streamlit code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def render_notebook_page() -> None:
    state = get_or_init_state()
    st.title("📓 Notebook Mode")

    if state.execution_backend == ExecutionBackend.COLAB_MCP:
        _render_colab_mcp_notebook(state)
    else:
        _render_local_notebook(state)


# ------------------------------------------------------------------
# Colab MCP notebook
# ------------------------------------------------------------------

def _render_colab_mcp_notebook(state) -> None:  # noqa: ANN001
    st.markdown(
        "Colab MCP connects your local agent to a **Google Colab** notebook in the "
        "cloud — no local GPU or heavy compute needed."
    )

    # --- Prerequisites check ---
    from app.backends.colab_mcp_backend import _find_uvx

    uvx_ok = _find_uvx() is not None
    try:
        from mcp import ClientSession  # noqa: F401
        mcp_ok = True
    except ImportError:
        mcp_ok = False

    if not uvx_ok or not mcp_ok:
        st.error("**Prerequisites missing**")
        if not uvx_ok:
            st.code("pip install uv", language="bash")
        if not mcp_ok:
            st.code("pip install 'mcp>=1.0'", language="bash")
        st.info("Install the above, then reload this page.")
        return

    st.success("Prerequisites OK (`uvx` and `mcp` SDK detected)")

    # --- Session management ---
    backend = st.session_state.get(_BACKEND_KEY)
    session_info = st.session_state.get(_SESSION_KEY)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚀 Connect to Colab", key="colab_connect"):
            with st.spinner("Starting Colab MCP server…"):
                backend = build_backend(ExecutionBackend.COLAB_MCP)
                session_info = _run_async(backend.prepare_session())
                st.session_state[_BACKEND_KEY] = backend
                st.session_state[_SESSION_KEY] = session_info

    with col2:
        if st.button("🔗 Open Browser Link", key="colab_open_browser", disabled=backend is None):
            if backend is not None:
                with st.spinner("Opening Colab in browser…"):
                    result = _run_async(backend.open_browser_connection())
                    if result.get("success"):
                        st.success("Browser connection initiated — check your browser!")
                    else:
                        st.warning(f"Connection attempt: {result.get('output', 'no details')}")

    with col3:
        if st.button("🔌 Disconnect", key="colab_disconnect", disabled=backend is None):
            if backend is not None:
                _run_async(backend.cleanup())
                st.session_state.pop(_BACKEND_KEY, None)
                st.session_state.pop(_SESSION_KEY, None)
                backend = None
                session_info = None
                st.info("Disconnected from Colab MCP.")

    # --- Status display ---
    if session_info:
        status = session_info.get("status", "unknown")
        if status == "ready":
            tools = session_info.get("tools", [])
            st.success(f"Connected — {len(tools)} tool(s) available")
            if tools:
                with st.expander("Available MCP Tools"):
                    for tool_name in tools:
                        st.code(tool_name, language="text")
        elif status == "error":
            st.error(f"Connection failed: {session_info.get('detail', 'unknown error')}")
        else:
            st.info(f"Session status: {status}")

    # --- Code execution ---
    st.divider()
    st.subheader("Execute Code in Colab")
    code = st.text_area(
        "Python code",
        height=200,
        placeholder="# Write code to execute in your Colab notebook…\nimport pandas as pd\nprint('Hello from Colab!')",
        key="colab_code_input",
    )

    can_execute = (
        backend is not None
        and session_info is not None
        and session_info.get("status") == "ready"
    )

    if st.button("▶️ Run in Colab", key="colab_run", disabled=not can_execute):
        if code.strip() and backend is not None:
            with st.spinner("Executing in Colab…"):
                # Try available execution tools — the proxied tools vary
                # depending on Colab frontend state.
                tools = _run_async(backend.list_tools())
                executed = False
                for tool_name in ("execute_cell", "run_code", "add_and_run_cell"):
                    if tool_name in tools:
                        result = _run_async(backend.run_job({
                            "tool": tool_name,
                            "arguments": {"code": code},
                        }))
                        st.code(result.get("output", "(no output)"), language="text")
                        executed = True
                        break
                if not executed:
                    st.warning(
                        "No execution tool found in the current session. "
                        "Make sure a Colab notebook is open and connected."
                    )

    # --- Refresh tools ---
    if can_execute and st.button("🔄 Refresh Tools", key="colab_refresh_tools"):
        tools = _run_async(backend.list_tools())
        st.info(f"Tools refreshed: {len(tools)} available")
        for t in tools:
            st.code(t, language="text")

    st.caption(
        f"Backend: **{state.execution_backend.value}** · "
        f"Provider: **{state.provider.value}**"
    )
    st.caption(
        "💡 Switch to **local** backend in Settings if you prefer local execution."
    )


# ------------------------------------------------------------------
# Local notebook (fallback)
# ------------------------------------------------------------------

def _render_local_notebook(state) -> None:  # noqa: ANN001
    st.info(
        "You are using the **local** execution backend. Notebook orchestration "
        "runs on your machine.  Switch to **colab_mcp** in Settings for "
        "cloud-based execution on Google Colab."
    )
    st.markdown(
        "### Local Notebook Capabilities\n"
        "- Inline cell execution using the local Python runtime\n"
        "- Kernel management tied to the local backend\n"
        "- Experiment tracking hooks via MLflow\n"
    )
    st.caption(
        f"Provider: **{state.provider.value}** · "
        f"Backend: **{state.execution_backend.value}**"
    )
