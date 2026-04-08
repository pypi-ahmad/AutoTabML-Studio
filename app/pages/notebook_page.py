"""Notebook page – browse auto-generated notebooks per dataset / job.

Each completed job (benchmark, experiment, flaml, profiling, validation) gets a
reproducible Jupyter notebook generated on demand.  The Colab MCP execution
backend is available in a collapsible section at the bottom.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from pathlib import Path

import streamlit as st

from app.backends import build_backend
from app.config.enums import ExecutionBackend
from app.notebooks.generator import generate_job_notebook
from app.pages.dataset_workspace import go_to_page
from app.pages.ui_labels import format_enum_value
from app.state.session import get_or_init_state
from app.storage import build_metadata_store
from app.storage.models import AppJobType

logger = logging.getLogger(__name__)

_BACKEND_KEY = "notebook_backend"
_SESSION_KEY = "notebook_session_info"

_NOTEBOOKS_DIR = Path("artifacts/notebooks")


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
    metadata_store = build_metadata_store(state.settings)
    st.title("📓 Notebooks")
    st.caption("Auto-generated Jupyter notebooks for every job you’ve run — ready to download, share, or open in Google Colab.")

    # ── Primary view: notebooks per dataset / job ────────────────────
    _render_job_notebooks(metadata_store)

    # ── Colab MCP / Local backend (collapsible) ──────────────────────
    st.divider()
    with st.expander("⚡ Advanced: Run Code in the Cloud or Locally", expanded=False):
        if state.execution_backend == ExecutionBackend.COLAB_MCP:
            _render_colab_mcp_notebook(state)
        else:
            _render_local_notebook(state)


# ------------------------------------------------------------------
# Job notebooks – one notebook per job per dataset
# ------------------------------------------------------------------

_JOB_TYPES_WITH_NOTEBOOKS = [
    AppJobType.BENCHMARK,
    AppJobType.EXPERIMENT,
    AppJobType.PROFILING,
    AppJobType.VALIDATION,
]

_JOB_ICONS = {
    AppJobType.BENCHMARK: "📊",
    AppJobType.EXPERIMENT: "🧪",
    AppJobType.PROFILING: "📋",
    AppJobType.VALIDATION: "✅",
}


def _render_job_notebooks(metadata_store) -> None:  # noqa: ANN001
    """Show notebooks grouped by dataset, one per job."""
    all_jobs = metadata_store.list_recent_jobs(limit=200)
    relevant = [j for j in all_jobs if j.job_type in _JOB_TYPES_WITH_NOTEBOOKS]

    if not relevant:
        st.info(
            "**No notebooks yet.**\n\n"
            "A Jupyter notebook is auto-generated every time you run a benchmark, experiment, or profiling job. "
            "You can download it, share it, or open it in Google Colab.\n\n"
            "**Next step:** Load a dataset and run your first workflow."
        )
        if st.button("📥 Load Data", key="nb_goto_load", type="primary"):
            go_to_page("Load Data")
        return

    # Group by dataset
    by_dataset: dict[str, list] = {}
    for job in relevant:
        key = job.dataset_name or "Unknown Dataset"
        by_dataset.setdefault(key, []).append(job)

    # Dataset selector
    dataset_names = sorted(by_dataset.keys())
    selected_dataset = st.selectbox(
        "Dataset",
        dataset_names,
        key="nb_dataset_select",
        help="Which dataset's jobs to generate a notebook from.",
    )
    if selected_dataset is None:
        return

    jobs = by_dataset[selected_dataset]
    st.markdown(f"### {selected_dataset}")
    st.caption(f"{len(jobs)} job(s)")

    for job in jobs:
        icon = _JOB_ICONS.get(job.job_type, "📄")
        label = f"{icon} **{format_enum_value(job.job_type.value)}** — {job.created_at.strftime('%Y-%m-%d %H:%M')}"
        meta = job.metadata or {}

        # Extra detail
        detail_parts = []
        if meta.get("best_model_name"):
            detail_parts.append(f"Best: {meta['best_model_name']}")
        if meta.get("best_score") is not None:
            detail_parts.append(f"Score: {meta['best_score']:.4f}")
        if meta.get("selected_model_name"):
            detail_parts.append(f"Model: {meta['selected_model_name']}")
        if detail_parts:
            label += f"  ·  {' · '.join(detail_parts)}"

        with st.expander(label, expanded=False):
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(
                    "📥 Generate & Download",
                    key=f"nb_gen_{job.job_id}",
                ):
                    nb_path = _generate_notebook_for_job(job)
                    st.session_state[f"nb_path_{job.job_id}"] = str(nb_path)
                    st.success(f"Notebook generated: `{nb_path.name}`")

            with col2:
                if st.button(
                    "👁️ Preview",
                    key=f"nb_preview_{job.job_id}",
                ):
                    nb_path = _generate_notebook_for_job(job)
                    st.session_state[f"nb_preview_data_{job.job_id}"] = nb_path.read_text(encoding="utf-8")

            # Download link
            cached_path = st.session_state.get(f"nb_path_{job.job_id}")
            if cached_path:
                p = Path(cached_path)
                if p.exists():
                    data = p.read_bytes()
                    st.download_button(
                        "⬇️ Download .ipynb",
                        data=data,
                        file_name=p.name,
                        mime="application/x-ipynb+json",
                        key=f"nb_dl_{job.job_id}",
                    )

            # Preview
            preview_data = st.session_state.get(f"nb_preview_data_{job.job_id}")
            if preview_data:
                _render_notebook_preview(preview_data)


def _generate_notebook_for_job(job) -> Path:  # noqa: ANN001
    """Generate and return notebook path for a job record."""
    meta = job.metadata or {}
    task_type = meta.get("task_type")
    target_column = meta.get("target_column")
    return generate_job_notebook(
        dataset_name=job.dataset_name or "unknown",
        job_type=job.job_type.value,
        task_type=task_type,
        target_column=target_column,
        metadata=meta,
        artifact_path=str(job.primary_artifact_path) if job.primary_artifact_path else None,
        summary_path=str(job.summary_path) if job.summary_path else None,
        output_dir=_NOTEBOOKS_DIR,
    )


def _render_notebook_preview(notebook_json: str) -> None:
    """Render a simple preview of notebook cells."""
    try:
        nb = json.loads(notebook_json)
    except json.JSONDecodeError:
        st.error("Could not parse notebook.")
        return

    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            st.markdown(source)
        elif cell["cell_type"] == "code":
            st.code(source, language="python")


# ------------------------------------------------------------------
# Colab MCP notebook
# ------------------------------------------------------------------

def _render_colab_mcp_notebook(state) -> None:  # noqa: ANN001
    st.markdown(
        "This connects your machine to a **Google Colab** notebook in the "
        "cloud — so you can run heavy computations without needing a powerful local machine."
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
        st.error(
            "**Cloud connection is not set up yet.** "
            "Some required packages need to be installed by your administrator."
        )
        with st.expander("Technical details for administrators"):
            st.caption("Run these commands in a terminal to install the required packages:")
            if not uvx_ok:
                st.code("pip install uv", language="bash")
            if not mcp_ok:
                st.code("pip install 'mcp>=1.0'", language="bash")
        st.info("Once the packages are installed, reload this page.")
        return

    st.success("All prerequisites are installed and ready.")

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
                with st.expander("Available cloud tools"):
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
        f"Running on: **{'Cloud (Google Colab)' if state.execution_backend.value == 'colab_mcp' else 'Local'}** · "
        f"AI Provider: **{state.provider.value}**"
    )
    st.caption(
        "💡 Switch to **Local** in Settings if you prefer to run everything on your own machine."
    )


# ------------------------------------------------------------------
# Local notebook (fallback)
# ------------------------------------------------------------------

def _render_local_notebook(state) -> None:  # noqa: ANN001
    st.info(
        "You’re running everything locally on your own machine. "
        "Switch to **Cloud (Google Colab)** in Settings to run in the cloud instead."
    )
    st.markdown(
        "### What You Can Do Locally\n"
        "- Run code cells using your machine’s Python\n"
        "- Track experiments automatically via MLflow\n"
    )
    st.caption(
        f"AI Provider: **{state.provider.value}** · "
        f"Running on: **Local**"
    )
