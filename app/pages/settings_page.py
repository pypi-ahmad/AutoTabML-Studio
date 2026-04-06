"""Settings / Runtime configuration page for Streamlit."""

from __future__ import annotations

import asyncio
import logging

import streamlit as st

from app.config.enums import (
    ExecutionBackend,
    LLMProvider,
    WorkspaceMode,
)
from app.gpu import cuda_summary
from app.providers.base import ModelItem
from app.providers.catalog_service import (
    build_provider,
    get_allowed_providers,
    resolve_default_model,
)
from app.security.masking import (
    MSG_DEFAULT_MODEL_UNAVAILABLE,
    MSG_EMPTY_OLLAMA_CATALOG,
    MSG_MISSING_API_KEY,
    MSG_MODEL_FETCH_FAILED,
    MSG_PROVIDER_UNREACHABLE,
    mask_secret,
)
from app.state.session import RuntimeState, get_or_init_state

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from sync Streamlit code.

    Streamlit may already have a running event loop, so ``asyncio.run()`` fails.
    The ThreadPoolExecutor workaround is the standard pattern — each invocation
    gets its own fresh event loop in a worker thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def render_settings_page() -> None:
    """Render the full Settings page."""
    st.title("⚙️ Settings")
    state: RuntimeState = get_or_init_state()

    _section_workspace(state)
    _section_execution(state)
    _section_accelerators(state)
    _section_provider(state)
    _section_credentials(state)
    _section_models(state)
    _section_save(state)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _section_workspace(state: RuntimeState) -> None:
    st.header("Workspace")
    modes = [m.value for m in WorkspaceMode]
    current_idx = modes.index(state.workspace_mode.value)
    selected = st.selectbox(
        "Workspace mode",
        options=modes,
        index=current_idx,
        key="ws_mode",
        help="**Dashboard** – interactive Streamlit UI.  **Notebook** – placeholder entry point for a future notebook workflow.",
    )
    state.workspace_mode = WorkspaceMode(selected)


def _section_execution(state: RuntimeState) -> None:
    st.header("Execution")
    backends = [b.value for b in ExecutionBackend]
    current_idx = backends.index(state.execution_backend.value)
    selected = st.selectbox(
        "Execution backend",
        options=backends,
        index=current_idx,
        key="exec_backend",
        help=(
            "**colab_mcp** (default) – execute notebooks and jobs on Google Colab "
            "via the Model Context Protocol.  "
            "**local** – run everything on your machine."
        ),
    )
    state.execution_backend = ExecutionBackend(selected)

    if state.execution_backend == ExecutionBackend.COLAB_MCP:
        from app.backends.colab_mcp_backend import _find_uvx

        uvx_ok = _find_uvx() is not None
        try:
            from mcp import ClientSession  # noqa: F401
            mcp_ok = True
        except ImportError:
            mcp_ok = False

        if uvx_ok and mcp_ok:
            st.success("Colab MCP prerequisites OK (`uvx` + `mcp` SDK detected)")
        else:
            missing = []
            if not uvx_ok:
                missing.append("`uv` (`pip install uv`)")
            if not mcp_ok:
                missing.append("`mcp` (`pip install 'mcp>=1.0'`)")
            st.warning(f"Colab MCP prerequisites missing: {', '.join(missing)}")

    elif state.execution_backend == ExecutionBackend.LOCAL:
        st.info(
            "Local backend selected — all compute runs on this machine. "
            "Switch to **colab_mcp** to offload to Google Colab."
        )


def _section_accelerators(state: RuntimeState) -> None:
    st.header("Accelerators")

    gpu_info = cuda_summary()
    if gpu_info["cuda_available"]:
        st.success(
            f"CUDA detected: {gpu_info['device_name'] or 'GPU available'}"
            f" (devices: {gpu_info['device_count']})"
        )
    else:
        st.caption("CUDA not detected in the current runtime. GPU-preferred experiment settings will fall back to CPU unless forced.")

    gpu_options: list[bool | str] = [True, False, "force"]
    selected = st.selectbox(
        "Default PyCaret GPU mode",
        options=gpu_options,
        index=gpu_options.index(state.settings.pycaret.default_use_gpu if state.settings.pycaret.default_use_gpu in gpu_options else True),
        format_func=lambda value: {
            True: "Prefer GPU when available",
            False: "CPU only",
            "force": "Require GPU and fail otherwise",
        }[value],
        key="pycaret_default_use_gpu",
        help="Applies to PyCaret experiment workflows in the UI and CLI.",
    )
    state.settings.pycaret.default_use_gpu = selected

    benchmark_prefer_gpu = st.checkbox(
        "Default benchmark GPU preference",
        value=bool(state.settings.benchmark.prefer_gpu),
        key="benchmark_prefer_gpu",
        help="Applies to LazyPredict benchmark workflows when CUDA and supported model libraries are available.",
    )
    state.settings.benchmark.prefer_gpu = benchmark_prefer_gpu


def _section_provider(state: RuntimeState) -> None:
    st.header("Provider")
    allowed = get_allowed_providers(state.execution_backend)
    allowed_values = [p.value for p in allowed]

    # If current provider is not allowed for this backend, reset
    if state.provider not in allowed:
        state.provider = allowed[0]

    current_idx = allowed_values.index(state.provider.value)
    selected = st.selectbox(
        "LLM provider",
        options=allowed_values,
        index=current_idx,
        key="llm_provider",
    )
    state.provider = LLMProvider(selected)

    if (
        state.execution_backend == ExecutionBackend.COLAB_MCP
        and LLMProvider.OLLAMA not in allowed
    ):
        st.caption("ℹ️ Ollama is disabled for the Colab MCP backend (local-only provider).")


def _section_credentials(state: RuntimeState) -> None:
    st.header("Credentials")
    provider = state.provider

    if provider in (LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GEMINI):
        label = f"{provider.value.capitalize()} API Key"
        current_raw = state.get_provider_api_key(provider) or ""
        masked_hint = mask_secret(current_raw) if current_raw else ""
        key_input = st.text_input(
            label,
            type="password",
            key=f"cred_{provider.value}",
            help=f"Your {provider.value} API key. Stored in-memory only – never written to disk.",
            placeholder=masked_hint or "Enter API key…",
        )
        if key_input:
            state.set_provider_api_key(provider, key_input)

    elif provider == LLMProvider.OLLAMA:
        url = st.text_input(
            "Ollama base URL",
            value=state.settings.ollama_base_url,
            key="ollama_url",
            help="URL of your local Ollama server.",
        )
        state.settings.ollama_base_url = url


def _section_models(state: RuntimeState) -> None:
    st.header("Models")

    fallback_default = state.settings.default_model_for_provider(state.provider)
    if fallback_default:
        st.caption(
            f"Verified fallback default for **{state.provider.value}**: `{fallback_default}`. "
            "This is only used when the live provider model catalog is unavailable."
        )

    if st.button("🔄 Refresh models", key="refresh_models"):
        _fetch_models(state)

    models = state.fetched_models
    if state.model_fetch_error:
        st.warning(state.model_fetch_error)

    if not models:
        st.info("No models loaded. Click **Refresh models** to fetch from the selected provider.")
        return

    model_ids = [m.id for m in models]
    display_map = {m.id: m.display_name for m in models}

    # Determine pre-selection
    pre_index = 0
    if state.selected_model_id and state.selected_model_id in model_ids:
        pre_index = model_ids.index(state.selected_model_id)
    else:
        default_item = resolve_default_model(models, state.provider)
        if default_item:
            pre_index = model_ids.index(default_item.id)

    selected_id = st.selectbox(
        "Model",
        options=model_ids,
        index=pre_index,
        format_func=lambda mid: display_map.get(mid, mid),
        key="model_select",
    )
    state.selected_model_id = selected_id
    st.caption(f"Model ID: `{selected_id}`")


def _section_save(state: RuntimeState) -> None:
    st.divider()
    if st.button("💾 Save settings", key="save_settings"):
        state.persist()
        st.success("Settings saved (secrets are NOT written to disk).")


# ---------------------------------------------------------------------------
# Model fetch helper
# ---------------------------------------------------------------------------

def _fetch_models(state: RuntimeState) -> None:
    """Build a provider instance and fetch models, updating state."""
    state.fetched_models = []
    state.model_fetch_error = None
    provider_enum = state.provider

    try:
        prov = build_provider(
            provider_enum,
            api_key=state.get_provider_api_key(provider_enum),
            base_url=state.settings.ollama_base_url if provider_enum == LLMProvider.OLLAMA else None,
        )
    except ValueError as exc:
        state.model_fetch_error = MSG_MISSING_API_KEY.format(provider=provider_enum.value)
        logger.warning("Provider build failed: %s", exc)
        return

    # Validate credentials / connectivity
    valid = _run_async(prov.validate_credentials())
    if not valid:
        state.model_fetch_error = MSG_PROVIDER_UNREACHABLE.format(provider=provider_enum.value)
        return

    # Fetch
    models: list[ModelItem] = _run_async(prov.list_models())
    if not models:
        if provider_enum == LLMProvider.OLLAMA:
            state.model_fetch_error = MSG_EMPTY_OLLAMA_CATALOG
        else:
            state.model_fetch_error = MSG_MODEL_FETCH_FAILED.format(
                provider=provider_enum.value, detail="Empty model list returned."
            )
        return

    state.fetched_models = models

    # Check default-model availability
    default_model_id = prov.get_default_model()
    default_item = resolve_default_model(models, provider_enum)
    if default_model_id and default_item and default_item.id != default_model_id:
        state.model_fetch_error = MSG_DEFAULT_MODEL_UNAVAILABLE.format(
            model_id=default_model_id,
            provider=provider_enum.value,
        )
    if default_item:
        state.selected_model_id = default_item.id
