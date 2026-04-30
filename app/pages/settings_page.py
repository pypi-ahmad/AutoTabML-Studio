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
from app.pages.ui_cache import (
    DATASET_LOAD_TTL_SECONDS,
    MLFLOW_QUERY_TTL_SECONDS,
    invalidate_all_ui_caches,
    invalidate_dataset_cache,
    invalidate_mlflow_query_cache,
    invalidate_service_cache,
)
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

    essentials_tab, advanced_tab = st.tabs(["Essentials", "Advanced"])

    with essentials_tab:
        _section_privacy_summary()
        _section_workspace(state)
        _section_accelerators_summary(state)
        _section_descriptions_toggle(state)
        _section_save(state, key_suffix="_essentials")

    with advanced_tab:
        _section_execution(state)
        _section_accelerators(state)
        _section_provider(state)
        _section_credentials(state)
        _section_models(state)
        _section_mlflow_descriptions(state)
        _section_cache_controls()
        _section_save(state, key_suffix="_advanced")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _section_privacy_summary() -> None:
    """Always-visible privacy reminder at the top of Essentials."""
    st.header("🔒 Privacy")
    st.info(
        "**Your data stays on your machine.** AutoTabML Studio is local-first — "
        "datasets, models, and results are stored in a local folder on your computer. "
        "Nothing is uploaded to external servers unless you explicitly configure a cloud backend."
    )


def _section_workspace(state: RuntimeState) -> None:
    st.header("Workspace")
    from app.pages.ui_labels import MODE_LABELS, make_format_func
    modes = [m.value for m in WorkspaceMode]
    current_idx = modes.index(state.workspace_mode.value)
    selected = st.selectbox(
        "Workspace mode",
        options=modes,
        index=current_idx,
        key="ws_mode",
        format_func=make_format_func(MODE_LABELS),
        help="**Dashboard** – interactive Streamlit UI.  **Notebook** – Jupyter-based workflow for advanced users.",
    )
    state.workspace_mode = WorkspaceMode(selected)


def _section_accelerators_summary(state: RuntimeState) -> None:
    """Lightweight GPU status for the Essentials tab — no controls."""
    st.header("GPU Status")
    gpu_info = cuda_summary()
    if gpu_info["cuda_available"]:
        st.success(
            f"⚡ GPU detected: **{gpu_info['device_name'] or 'available'}** "
            f"({gpu_info['device_count']} device(s)). Training will be faster."
        )
    else:
        st.info(
            "💻 No GPU detected — training runs on CPU. "
            "For GPU options, see the **Advanced** tab."
        )


def _section_descriptions_toggle(state: RuntimeState) -> None:
    """Simple on/off for run descriptions on the Essentials tab."""
    st.header("Run Summaries")
    desc_enabled = st.checkbox(
        "Auto-generate a plain-English summary for every workflow run",
        value=state.settings.mlflow_descriptions_enabled,
        key="essentials_desc_toggle",
        help=(
            "After each benchmark, experiment, or prediction, the app writes a short summary "
            "explaining what happened and what to do next. Fine-tune AI options in the **Advanced** tab."
        ),
    )
    state.settings.mlflow_descriptions_enabled = desc_enabled
    if not desc_enabled:
        state.settings.llm_descriptions_enabled = False


def _section_execution(state: RuntimeState) -> None:
    st.header("Where to Run")
    from app.pages.ui_labels import BACKEND_LABELS, make_format_func
    backends = [b.value for b in ExecutionBackend]
    current_idx = backends.index(state.execution_backend.value)
    selected = st.selectbox(
        "Execution environment",
        options=backends,
        index=current_idx,
        format_func=make_format_func(BACKEND_LABELS, fallback_title=False),
        key="exec_backend",
        help=(
            "**Cloud (Google Colab)** — run heavy computations on Google’s free cloud GPUs.  "
            "**Local** — run everything on your own machine."
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
            st.success("Cloud connection ready \u2714")
        else:
            st.warning(
                "Cloud connection not ready — some required packages are missing. "
                "Ask your administrator to install the cloud connection prerequisites."
            )

    elif state.execution_backend == ExecutionBackend.LOCAL:
        st.info(
            "Local backend selected — all compute runs on this machine. "
            "Switch to **Cloud (Google Colab)** to offload heavy tasks."
        )


def _section_accelerators(state: RuntimeState) -> None:
    st.header("GPU Acceleration")

    gpu_info = cuda_summary()
    if gpu_info["cuda_available"]:
        st.success(
            f"⚡ GPU detected: {gpu_info['device_name'] or 'available'}"
            f" ({gpu_info['device_count']} device(s)). Training will be faster."
        )
    else:
        st.caption("💻 No GPU detected. The app will use your CPU — training may be slower for large datasets.")

    gpu_options: list[bool | str] = [True, False, "force"]
    selected = st.selectbox(
        "GPU mode for experiments",
        options=gpu_options,
        index=gpu_options.index(state.settings.pycaret.default_use_gpu if state.settings.pycaret.default_use_gpu in gpu_options else True),
        format_func=lambda value: {
            True: "Use GPU when available (recommended)",
            False: "CPU only",
            "force": "Require GPU (stop if unavailable)",
        }[value],
        key="pycaret_default_use_gpu",
        help="Controls whether model training uses GPU acceleration when a compatible GPU is detected.",
    )
    state.settings.pycaret.default_use_gpu = selected

    benchmark_prefer_gpu = st.checkbox(
        "Use GPU for benchmarks when available",
        value=bool(state.settings.benchmark.prefer_gpu),
        key="benchmark_prefer_gpu",
        help="Speeds up algorithm comparison by using GPU-accelerated libraries when possible.",
    )
    state.settings.benchmark.prefer_gpu = benchmark_prefer_gpu

    # ── FLAML defaults ─────────────────────────────────────────────────
    st.subheader("FLAML AutoML Defaults")
    flaml_time_budget = int(st.number_input(
        "Default time budget (seconds)",
        min_value=10,
        max_value=3600,
        value=int(state.settings.flaml.default_time_budget),
        step=10,
        key="flaml_default_time_budget",
        help="How long FLAML searches for the best model by default.",
    ))
    state.settings.flaml.default_time_budget = flaml_time_budget

    flaml_n_splits = int(st.number_input(
        "Default cross-validation folds",
        min_value=2,
        max_value=20,
        value=int(state.settings.flaml.default_n_splits),
        step=1,
        key="flaml_default_n_splits",
        help="Number of CV folds for FLAML evaluation.",
    ))
    state.settings.flaml.default_n_splits = flaml_n_splits


def _section_provider(state: RuntimeState) -> None:
    st.header("AI Provider")
    st.caption("Choose which AI service powers descriptions and optional smart features.")
    from app.pages.ui_labels import PROVIDER_LABELS, make_format_func
    allowed = get_allowed_providers(state.execution_backend)
    allowed_values = [p.value for p in allowed]

    # If current provider is not allowed for this backend, reset
    if state.provider not in allowed:
        state.provider = allowed[0]

    current_idx = allowed_values.index(state.provider.value)
    selected = st.selectbox(
        "AI service",
        options=allowed_values,
        index=current_idx,
        key="llm_provider",
        format_func=make_format_func(PROVIDER_LABELS),
        help="Select which AI service to use for generating descriptions and smart features.",
    )
    state.provider = LLMProvider(selected)

    if (
        state.execution_backend == ExecutionBackend.COLAB_MCP
        and LLMProvider.OLLAMA not in allowed
    ):
        st.caption("ℹ️ Ollama is not available when running in the cloud (it requires a local server).")


def _section_credentials(state: RuntimeState) -> None:
    st.header("API Keys")
    st.caption(
        "🔒 Keys are stored in memory only — they are never saved to disk "
        "and disappear when you close the app."
    )
    provider = state.provider

    if provider in (LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GEMINI):
        from app.pages.ui_labels import PROVIDER_LABELS
        provider_name = PROVIDER_LABELS.get(provider.value, provider.value.title())
        current_raw = state.get_provider_api_key(provider) or ""
        masked_hint = mask_secret(current_raw) if current_raw else ""
        key_input = st.text_input(
            f"{provider_name} API Key",
            type="password",
            key=f"cred_{provider.value}",
            help=f"Your {provider_name} API key. Needed for AI-powered summaries.",
            placeholder=masked_hint or "Enter API key…",
        )
        if key_input:
            state.set_provider_api_key(provider, key_input)

    elif provider == LLMProvider.OLLAMA:
        url = st.text_input(
            "Ollama server address",
            value=state.settings.ollama_base_url,
            key="ollama_url",
            help="URL of your local Ollama server (e.g. http://localhost:11434).",
        )
        state.settings.ollama_base_url = url


def _section_models(state: RuntimeState) -> None:
    st.header("AI Model")
    st.caption("Which AI model to use for generating run summaries and smart features.")

    fallback_default = state.settings.default_model_for_provider(state.provider)
    if fallback_default:
        st.caption(f"Default: **{fallback_default}**")

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


def _section_mlflow_descriptions(state: RuntimeState) -> None:
    st.divider()
    st.subheader("📝 Run Summaries")
    st.caption(
        "Automatically write a plain-English summary after every workflow run — "
        "covering what happened, key results, and suggested next steps."
    )

    desc_enabled = st.checkbox(
        "Generate a summary for every run",
        value=state.settings.mlflow_descriptions_enabled,
        key="mlflow_desc_enabled",
        help="A human-readable summary is created for each job run "
        "and shown in the History page.",
    )
    state.settings.mlflow_descriptions_enabled = desc_enabled

    if desc_enabled:
        llm_enabled = st.checkbox(
            "Use AI for richer summaries (requires provider + API key above)",
            value=state.settings.llm_descriptions_enabled,
            key="llm_desc_enabled",
            help="When enabled, the AI service writes more insightful, "
            "narrative-style summaries. When off, built-in templates are used.",
        )
        state.settings.llm_descriptions_enabled = llm_enabled

        if llm_enabled:
            from app.pages.ui_labels import PROVIDER_LABELS
            provider_name = PROVIDER_LABELS.get(state.provider.value, state.provider.value.title())
            api_key = state.get_provider_api_key(state.provider)
            has_key = bool(api_key)
            model_id = state.selected_model_id

            if has_key and model_id:
                st.success(
                    f"AI summaries will use **{provider_name}** "
                    f"(model: **{model_id}**)."
                )
            elif has_key:
                st.warning(
                    f"**{provider_name}** API key is set but no model selected. "
                    "Click **Refresh models** above to pick one."
                )
            else:
                st.warning(
                    f"**{provider_name}** API key is needed for AI summaries. "
                    "Enter it in the **API Keys** section above."
                )
    else:
        state.settings.llm_descriptions_enabled = False


def _section_cache_controls() -> None:
    st.divider()
    st.subheader("Performance Cache")
    st.caption(
        f"MLflow query data is cached for **{MLFLOW_QUERY_TTL_SECONDS} seconds** and dataset loads for **{DATASET_LOAD_TTL_SECONDS // 60} minutes**. "
        "Use these controls to force-refresh UI caches when source files or tracking state change outside the current page."
    )

    cache_col1, cache_col2, cache_col3 = st.columns(3)
    if cache_col1.button("Clear MLflow Cache", key="clear_mlflow_cache"):
        invalidate_mlflow_query_cache()
        st.success("Cleared cached MLflow query data.")

    if cache_col2.button("Clear Dataset Cache", key="clear_dataset_cache"):
        invalidate_dataset_cache()
        st.success("Cleared cached dataset loads.")

    if cache_col3.button("Clear Service Cache", key="clear_service_cache"):
        invalidate_service_cache()
        st.success("Cleared cached UI services and metadata resources.")

    if st.button("Clear All UI Caches", key="clear_all_ui_caches"):
        invalidate_all_ui_caches()
        st.success("Cleared all Streamlit UI caches.")


def _section_save(state: RuntimeState, *, key_suffix: str = "") -> None:
    st.divider()
    if st.button("💾 Save settings", key=f"save_settings{key_suffix}"):
        state.persist()
        st.success("Settings saved. API keys are kept in memory only and never written to disk.")


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
