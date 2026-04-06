"""Shared Streamlit dataset workspace helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from app.artifacts import LocalArtifactManager
from app.ingestion import DatasetInputSpec, IngestionSourceType, LoadedDataset, load_dataset
from app.ingestion.types import DELIMITED_FILE_SUFFIXES, EXCEL_FILE_SUFFIXES
from app.ingestion.uci_loader import list_available_uci_datasets
from app.security.masking import safe_error_message
from app.state.session import get_or_init_state
from app.storage import ProjectRecord, build_metadata_store, ensure_dataset_record

_LOADED_DATASETS_KEY = "loaded_datasets"
_ACTIVE_DATASET_NAME_KEY = "active_dataset_name"
_WORKFLOW_RESULT_KEYS = [
    "validation_summaries",
    "validation_bundles",
    "profiling_summaries",
    "profiling_bundles",
    "benchmark_bundles",
    "experiment_bundles",
]
_SUPPORTED_UPLOAD_TYPES = ["csv", "tsv", "txt", "data", "xlsx", "xls", "xlsm", "xlsb"]
_ACTIVE_DATASET_METADATA_KEYS = (
    "active_dataset_name",
    "active_dataset_key",
    "active_dataset_source_locator",
    "active_dataset_selected_at",
)
_UCI_CATALOG_RESULTS_KEY_TEMPLATE = "{key_prefix}_uci_catalog_results"


def infer_local_source_type(file_name: str) -> IngestionSourceType:
    """Infer the ingestion source type from a local filename or path."""

    suffix = Path(file_name).suffix.lower()
    if suffix in EXCEL_FILE_SUFFIXES:
        return IngestionSourceType.EXCEL
    if suffix == ".csv":
        return IngestionSourceType.CSV
    if suffix in DELIMITED_FILE_SUFFIXES:
        return IngestionSourceType.DELIMITED_TEXT
    raise ValueError(f"Unsupported dataset file type: '{suffix or '(none)'}'.")


def build_local_path_input_spec(path_value: str, *, display_name: str | None = None) -> DatasetInputSpec:
    """Build an ingestion input spec for a local file path."""

    cleaned = path_value.strip()
    if not cleaned:
        raise ValueError("Provide a local dataset path.")
    path = Path(cleaned)
    return DatasetInputSpec(
        source_type=infer_local_source_type(path.name),
        path=path,
        display_name=display_name or None,
    )


def build_url_input_spec(url_value: str, *, display_name: str | None = None) -> DatasetInputSpec:
    """Build an ingestion input spec for a remote URL."""

    cleaned = url_value.strip()
    if not cleaned:
        raise ValueError("Provide a dataset URL.")
    return DatasetInputSpec(
        source_type=IngestionSourceType.URL_FILE,
        url=cleaned,
        display_name=display_name or None,
    )


def get_loaded_datasets() -> dict[str, LoadedDataset]:
    """Return the current session datasets."""

    return st.session_state.setdefault(_LOADED_DATASETS_KEY, {})


def get_active_loaded_dataset(*, metadata_store=None) -> tuple[str | None, LoadedDataset | None]:  # noqa: ANN001
    """Return the currently active session dataset, if available."""

    loaded = get_loaded_datasets()
    active_name = get_active_dataset_name(metadata_store=metadata_store)
    if active_name is None:
        return None, None
    return active_name, loaded.get(active_name)


def get_active_dataset_name(*, metadata_store=None) -> str | None:  # noqa: ANN001
    """Resolve the current active dataset name from session or persisted workspace metadata."""

    loaded = get_loaded_datasets()
    active_name = _normalize_dataset_name(st.session_state.get(_ACTIVE_DATASET_NAME_KEY))
    if active_name in loaded:
        return active_name

    persisted_name, persisted_key = _read_persisted_active_dataset_selection(metadata_store)
    if persisted_key:
        for candidate_name, loaded_dataset in loaded.items():
            if _dataset_identity_key(loaded_dataset) == persisted_key:
                st.session_state[_ACTIVE_DATASET_NAME_KEY] = candidate_name
                return candidate_name

    if persisted_name in loaded:
        st.session_state[_ACTIVE_DATASET_NAME_KEY] = persisted_name
        return persisted_name

    if loaded:
        fallback_name = next(reversed(loaded))
        st.session_state[_ACTIVE_DATASET_NAME_KEY] = fallback_name
        return fallback_name

    st.session_state.pop(_ACTIVE_DATASET_NAME_KEY, None)
    return None


def set_active_dataset(
    dataset_name: str | None,
    *,
    metadata_store,
    loaded_dataset: LoadedDataset | None = None,
) -> str | None:  # noqa: ANN001
    """Persist the active dataset selection in session state and workspace metadata."""

    if not dataset_name:
        st.session_state.pop(_ACTIVE_DATASET_NAME_KEY, None)
        _persist_active_dataset_selection(metadata_store, dataset_name=None, loaded_dataset=None)
        return None

    loaded = get_loaded_datasets()
    selected_dataset = loaded_dataset or loaded.get(dataset_name)
    if selected_dataset is None:
        raise KeyError(f"Unknown loaded dataset: {dataset_name}")

    st.session_state[_ACTIVE_DATASET_NAME_KEY] = dataset_name
    ensure_dataset_record(metadata_store, selected_dataset, dataset_name=dataset_name)
    _persist_active_dataset_selection(metadata_store, dataset_name=dataset_name, loaded_dataset=selected_dataset)
    return dataset_name


def go_to_page(page_label: str) -> None:
    """Navigate to another registered page within the Streamlit app."""

    st.session_state["nav"] = page_label
    st.rerun()


def render_dataset_header(
    workflow_label: str,
    *,
    key_prefix: str,
    metadata_store=None,  # noqa: ANN001
) -> tuple[str | None, LoadedDataset | None]:
    """Unified dataset selection header for workflow pages.

    When an active dataset exists the header renders a compact info bar.
    When no dataset is loaded yet an inline quick-loader is displayed so the
    user can upload a file or enter a local path *without* leaving the page.

    Returns ``(active_name, LoadedDataset)`` or ``(None, None)``.
    """

    if metadata_store is None:
        metadata_store = build_metadata_store(get_or_init_state().settings)

    loaded = get_loaded_datasets()
    active_name = get_active_dataset_name(metadata_store=metadata_store)
    active_ds = loaded.get(active_name) if active_name else None

    if active_ds is not None:
        _render_dataset_info_bar(active_name, active_ds, loaded, key_prefix=key_prefix)
        return active_name, active_ds

    if loaded:
        first = next(iter(loaded))
        set_active_dataset(first, metadata_store=metadata_store)
        st.rerun()

    _render_inline_dataset_loader(workflow_label, key_prefix=key_prefix, metadata_store=metadata_store)
    return None, None


def _render_dataset_info_bar(
    active_name: str,
    active_ds: LoadedDataset,
    loaded: dict[str, LoadedDataset],
    *,
    key_prefix: str,
) -> None:
    """Compact banner showing active dataset stats."""

    df = active_ds.dataframe
    source = active_ds.metadata.source_type.value
    n_loaded = len(loaded)
    extra = f"  ·  {n_loaded} datasets in session" if n_loaded > 1 else ""

    col_info, col_action = st.columns([6, 1])
    col_info.caption(
        f"📋 **{active_name}**  ·  {len(df):,} rows × {len(df.columns)} cols  ·  {source}{extra}"
    )
    if col_action.button("📂", key=f"{key_prefix}_open_intake", help="Open full Dataset Intake"):
        go_to_page("Dataset Intake")


def _render_inline_dataset_loader(
    workflow_label: str,
    *,
    key_prefix: str,
    metadata_store,  # noqa: ANN001
) -> None:
    """Inline dataset loader shown when no dataset is active."""

    st.info(f"Load a dataset to start {workflow_label.lower()}.")

    upload_tab, path_tab = st.tabs(["📁 Upload File", "📂 Local Path"])

    with upload_tab:
        uploaded = st.file_uploader(
            "CSV, delimited text, or Excel",
            type=_SUPPORTED_UPLOAD_TYPES,
            key=f"{key_prefix}_gw_file",
        )
        if st.button("Load", key=f"{key_prefix}_gw_upload_btn", disabled=uploaded is None):
            if uploaded is not None:
                name = _load_into_session(
                    uploaded_file_to_input_spec(uploaded),
                    preferred_name=Path(uploaded.name).stem,
                    metadata_store=metadata_store,
                )
                if name:
                    st.rerun()

    with path_tab:
        local = st.text_input(
            "File path",
            key=f"{key_prefix}_gw_path",
            placeholder=r"datasets/sklearn/Diabetes/diabetes.csv",
        )
        if st.button("Load", key=f"{key_prefix}_gw_path_btn"):
            if local.strip():
                try:
                    spec = build_local_path_input_spec(local)
                except Exception as exc:
                    st.error(safe_error_message(exc))
                else:
                    name = _load_into_session(
                        spec,
                        preferred_name=Path(local).stem,
                        metadata_store=metadata_store,
                    )
                    if name:
                        st.rerun()
            else:
                st.error("Enter a file path.")

    st.caption("For URL, UCI Repository, or advanced options:")
    if st.button("Open Dataset Intake", key=f"{key_prefix}_gw_intake"):
        go_to_page("Dataset Intake")


def render_sidebar_dataset_status() -> None:
    """Render compact dataset status and switcher in the Streamlit sidebar."""

    state = get_or_init_state()
    metadata_store = build_metadata_store(state.settings)
    loaded = get_loaded_datasets()
    active_name = get_active_dataset_name(metadata_store=metadata_store)

    if not loaded or not active_name:
        st.sidebar.caption("📋 No dataset loaded")
        return

    st.sidebar.divider()

    if len(loaded) > 1:
        options = list(loaded.keys())
        idx = options.index(active_name) if active_name in options else 0
        new_name = st.sidebar.selectbox(
            "📋 Active dataset",
            options=options,
            index=idx,
            key="sidebar_ds_pick",
        )
        if new_name != active_name:
            set_active_dataset(new_name, metadata_store=metadata_store)
            st.rerun()
        ds = loaded[new_name]
        st.sidebar.caption(f"{len(ds.dataframe):,} rows × {len(ds.dataframe.columns)} cols")
    else:
        ds = loaded[active_name]
        st.sidebar.caption(
            f"📋 **{active_name}**  \n{len(ds.dataframe):,} rows × {len(ds.dataframe.columns)} cols"
        )


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------


def render_dataset_gateway_notice(workflow_label: str, *, key_prefix: str) -> None:
    """Render the standard no-active-dataset gateway notice.

    .. deprecated:: Use :func:`render_dataset_header` for inline loading.
    """

    st.warning(
        f"Select an active dataset on Dataset Intake before running {workflow_label.lower()}."
    )
    if st.button("Open Dataset Intake", key=f"{key_prefix}_open_dataset_intake"):
        go_to_page("Dataset Intake")


def render_active_dataset_banner(dataset_name: str, *, key_prefix: str) -> None:
    """Render the shared active-dataset banner used on workflow pages.

    .. deprecated:: Use :func:`render_dataset_header` for inline loading.
    """

    info_col, action_col = st.columns([5, 2])
    info_col.info(
        f"Using active dataset '{dataset_name}'. Change the selection on Dataset Intake if you need a different source."
    )
    if action_col.button("Dataset Intake", key=f"{key_prefix}_change_dataset"):
        go_to_page("Dataset Intake")


def uploaded_file_to_input_spec(uploaded_file) -> DatasetInputSpec:  # noqa: ANN001
    """Persist one uploaded file to the app temp area and return an input spec."""

    state = get_or_init_state()
    source_type = infer_local_source_type(uploaded_file.name)
    manager = LocalArtifactManager(state.settings.artifacts)
    suffix = Path(uploaded_file.name).suffix.lower()
    temp_path = manager.create_temp_file_path(stem=uploaded_file.name, suffix=suffix)
    manager.write_bytes(temp_path, uploaded_file.getbuffer())
    return DatasetInputSpec(
        source_type=source_type,
        path=temp_path,
        display_name=uploaded_file.name,
    )


def resolve_session_dataset_name(
    preferred_name: str | None,
    loaded_dataset: LoadedDataset,
    existing_names: list[str],
) -> str:
    """Return a stable, unique session dataset label."""

    base_name = _normalize_dataset_name(preferred_name)
    if not base_name:
        metadata_name = _normalize_dataset_name(loaded_dataset.metadata.display_name)
        if metadata_name:
            base_name = metadata_name
        elif loaded_dataset.input_spec and loaded_dataset.input_spec.path is not None:
            base_name = loaded_dataset.input_spec.path.stem
        elif loaded_dataset.input_spec and loaded_dataset.input_spec.url is not None:
            parsed = urlparse(loaded_dataset.input_spec.url)
            base_name = Path(parsed.path).stem or parsed.netloc or "dataset"
        else:
            base_name = "dataset"

    candidate = base_name
    counter = 2
    while candidate in existing_names:
        candidate = f"{base_name} ({counter})"
        counter += 1
    return candidate


def render_dataset_workspace(
    *,
    title: str | None = "Dataset Workspace",
    caption: str | None = None,
    key_prefix: str = "dataset_workspace",
) -> dict[str, LoadedDataset]:
    """Render the shared dataset loading and selection UI."""

    state = get_or_init_state()
    metadata_store = build_metadata_store(state.settings)
    loaded = get_loaded_datasets()

    if title:
        st.subheader(title)
    st.caption(caption or "Load a local file, workspace path, or URL into the current Streamlit session.")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Session datasets", len(loaded))
    if loaded:
        latest_name = next(reversed(loaded))
        latest_dataset = loaded[latest_name]
        metric_col2.metric("Latest rows", len(latest_dataset.dataframe))
        metric_col3.metric("Latest columns", len(latest_dataset.dataframe.columns))
    else:
        metric_col2.metric("Latest rows", 0)
        metric_col3.metric("Latest columns", 0)

    upload_tab, path_tab, url_tab, uci_tab, loaded_tab = st.tabs(["Upload", "Local Path", "URL", "UCI Repository", "Loaded"])

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload CSV, delimited text, or Excel",
            type=_SUPPORTED_UPLOAD_TYPES,
            key=f"{key_prefix}_upload_file",
        )
        upload_name = st.text_input(
            "Session dataset name (optional)",
            key=f"{key_prefix}_upload_name",
            help="Leave blank to use the file name.",
        ).strip()
        if st.button(
            "Load Uploaded Dataset",
            key=f"{key_prefix}_upload_button",
            disabled=uploaded_file is None,
        ):
            if uploaded_file is None:
                st.error("Choose a file before loading it.")
            else:
                _load_into_session(
                    uploaded_file_to_input_spec(uploaded_file),
                    preferred_name=upload_name or Path(uploaded_file.name).stem,
                    metadata_store=metadata_store,
                )
        if uploaded_file is not None:
            st.caption(f"Ready to load: {uploaded_file.name}")

    with path_tab:
        local_path = st.text_input(
            "Local dataset path",
            key=f"{key_prefix}_path_value",
            placeholder=r"artifacts\e2e\real_flow_20260404\iris_train.csv",
        ).strip()
        path_name = st.text_input(
            "Session dataset name (optional)",
            key=f"{key_prefix}_path_name",
        ).strip()
        if st.button("Load Local Path", key=f"{key_prefix}_path_button"):
            try:
                spec = build_local_path_input_spec(local_path, display_name=path_name or None)
            except Exception as exc:
                st.error(safe_error_message(exc))
            else:
                _load_into_session(
                    spec,
                    preferred_name=path_name or Path(local_path).stem,
                    metadata_store=metadata_store,
                )

    with url_tab:
        url_value = st.text_input(
            "Dataset URL",
            key=f"{key_prefix}_url_value",
            placeholder="https://example.com/data.csv",
        ).strip()
        url_name = st.text_input(
            "Session dataset name (optional)",
            key=f"{key_prefix}_url_name",
        ).strip()
        if st.button("Load URL", key=f"{key_prefix}_url_button"):
            try:
                spec = build_url_input_spec(url_value, display_name=url_name or None)
            except Exception as exc:
                st.error(safe_error_message(exc))
            else:
                parsed = urlparse(url_value)
                fallback_name = Path(parsed.path).stem or parsed.netloc or "dataset"
                _load_into_session(
                    spec,
                    preferred_name=url_name or fallback_name,
                    metadata_store=metadata_store,
                )

    with uci_tab:
        st.caption(
            "Load datasets directly from the [UCI Machine Learning Repository](https://archive.ics.uci.edu). "
            "Provide either a dataset ID or a dataset name."
        )
        catalog_results_key = _UCI_CATALOG_RESULTS_KEY_TEMPLATE.format(key_prefix=key_prefix)
        browse_col1, browse_col2, browse_col3 = st.columns(3)
        catalog_search = browse_col1.text_input(
            "Catalog search (optional)",
            key=f"{key_prefix}_uci_catalog_search",
            placeholder="e.g. iris, heart, wine",
            help="Uses ucimlrepo.list_available_datasets(search=...).",
        ).strip()
        catalog_area = browse_col2.text_input(
            "Catalog area (optional)",
            key=f"{key_prefix}_uci_catalog_area",
            placeholder="e.g. life science",
            help="Optional UCI area filter.",
        ).strip()
        catalog_filter = browse_col3.text_input(
            "Catalog filter (optional)",
            key=f"{key_prefix}_uci_catalog_filter",
            placeholder="e.g. aim-ahead",
            help="Optional ucimlrepo filter value.",
        ).strip()
        if st.button("Search UCI Catalog", key=f"{key_prefix}_uci_catalog_button"):
            try:
                results = list_available_uci_datasets(
                    search=catalog_search or None,
                    area=catalog_area or None,
                    filter=catalog_filter or None,
                )
            except Exception as exc:
                st.error(f"UCI catalog search failed: {safe_error_message(exc)}")
            else:
                st.session_state[catalog_results_key] = results
                if not results:
                    st.info("No matching UCI datasets found.")

        catalog_results = st.session_state.get(catalog_results_key, [])
        if catalog_results:
            st.dataframe(catalog_results, width="stretch")
            option_map = {
                f"{row['uci_id']} - {row['name']}": row
                for row in catalog_results
            }
            selected_catalog_item = st.selectbox(
                "Catalog result",
                options=list(option_map.keys()),
                key=f"{key_prefix}_uci_catalog_pick",
            )
            if st.button("Use Selected Dataset", key=f"{key_prefix}_uci_catalog_use"):
                selected_row = option_map[selected_catalog_item]
                st.session_state[f"{key_prefix}_uci_id"] = int(selected_row["uci_id"])
                st.session_state[f"{key_prefix}_uci_name"] = ""
                st.session_state[f"{key_prefix}_uci_display_name"] = selected_row["name"]
                st.rerun()

        st.divider()
        uci_col1, uci_col2 = st.columns(2)
        with uci_col1:
            uci_id_value = st.number_input(
                "Dataset ID",
                min_value=1,
                value=None,
                step=1,
                key=f"{key_prefix}_uci_id",
                help="Numeric ID from the UCI ML Repository (e.g. 53 for Iris, 45 for Heart Disease).",
                placeholder="e.g. 53",
            )
        with uci_col2:
            uci_name_value = st.text_input(
                "Dataset name (alternative to ID)",
                key=f"{key_prefix}_uci_name",
                placeholder="e.g. Iris",
                help="Full or partial dataset name. Only used when ID is not provided.",
            ).strip()
        uci_display_name = st.text_input(
            "Session dataset name (optional)",
            key=f"{key_prefix}_uci_display_name",
        ).strip()
        if st.button("Load UCI Dataset", key=f"{key_prefix}_uci_button"):
            resolved_id = int(uci_id_value) if uci_id_value is not None else None
            resolved_name = uci_name_value or None
            if resolved_id is None and not resolved_name:
                st.error("Provide a UCI dataset ID or name.")
            elif resolved_id is not None and resolved_name:
                st.error("Provide either a UCI dataset ID or a name, not both.")
            else:
                try:
                    spec = DatasetInputSpec(
                        source_type=IngestionSourceType.UCI_REPO,
                        uci_id=resolved_id,
                        uci_name=resolved_name,
                        display_name=uci_display_name or None,
                    )
                except Exception as exc:
                    st.error(safe_error_message(exc))
                else:
                    fallback = uci_display_name or (resolved_name if resolved_name else f"uci-{resolved_id}")
                    _load_into_session(
                        spec,
                        preferred_name=fallback,
                        metadata_store=metadata_store,
                    )

    with loaded_tab:
        if not loaded:
            st.caption("No datasets are loaded in the current Streamlit session.")
        else:
            active_name = get_active_dataset_name(metadata_store=metadata_store)
            options = list(loaded.keys())
            selected_name = st.selectbox(
                "Loaded dataset",
                options=options,
                index=options.index(active_name) if active_name in options else 0,
                key=f"{key_prefix}_loaded_dataset",
            )
            if selected_name != active_name:
                set_active_dataset(selected_name, metadata_store=metadata_store)
            selected_dataset = loaded[selected_name]
            st.caption(
                f"Active dataset: **{selected_name}** · Rows: **{len(selected_dataset.dataframe):,}** · Columns: **{len(selected_dataset.dataframe.columns):,}**"
            )
            st.dataframe(selected_dataset.preview(20), width="stretch")
            with st.expander("Dataset metadata"):
                st.json(selected_dataset.metadata.model_dump(mode="json"))

            action_col1, action_col2 = st.columns(2)
            if action_col1.button("Remove Selected Dataset", key=f"{key_prefix}_remove_button"):
                remove_loaded_dataset(selected_name, metadata_store=metadata_store)
                st.success(f"Removed '{selected_name}' from this session.")
            if action_col2.button("Clear Session Datasets", key=f"{key_prefix}_clear_button"):
                clear_loaded_datasets(metadata_store=metadata_store)
                st.success("Cleared all session datasets.")

    return get_loaded_datasets()


def remove_loaded_dataset(dataset_name: str, *, metadata_store) -> None:  # noqa: ANN001
    """Remove one dataset from the current session and repair active selection."""

    loaded = get_loaded_datasets()
    loaded.pop(dataset_name, None)
    _clear_dataset_results(dataset_name)

    if st.session_state.get(_ACTIVE_DATASET_NAME_KEY) == dataset_name:
        fallback_name = next(reversed(loaded)) if loaded else None
        set_active_dataset(fallback_name, metadata_store=metadata_store)
    elif not loaded:
        set_active_dataset(None, metadata_store=metadata_store)


def clear_loaded_datasets(*, metadata_store) -> None:  # noqa: ANN001
    """Clear all session datasets and any active selection."""

    st.session_state[_LOADED_DATASETS_KEY] = {}
    _clear_dataset_results()
    set_active_dataset(None, metadata_store=metadata_store)


def _load_into_session(
    input_spec: DatasetInputSpec,
    *,
    preferred_name: str | None,
    metadata_store,
) -> str | None:  # noqa: ANN001
    try:
        loaded_dataset = load_dataset(input_spec)
    except Exception as exc:
        st.error(f"Dataset load failed: {safe_error_message(exc)}")
        return None

    loaded = st.session_state.setdefault(_LOADED_DATASETS_KEY, {})
    dataset_name = resolve_session_dataset_name(preferred_name, loaded_dataset, list(loaded.keys()))
    loaded[dataset_name] = loaded_dataset
    set_active_dataset(dataset_name, metadata_store=metadata_store, loaded_dataset=loaded_dataset)
    st.success(f"Loaded dataset '{dataset_name}'.")
    return dataset_name


def _dataset_identity_key(loaded_dataset: LoadedDataset) -> str:
    metadata = loaded_dataset.metadata
    return str(metadata.content_hash or metadata.schema_hash or metadata.source_locator)


def _persist_active_dataset_selection(
    metadata_store,
    *,
    dataset_name: str | None,
    loaded_dataset: LoadedDataset | None,
) -> None:  # noqa: ANN001
    if metadata_store is None:
        return

    project = metadata_store.get_workspace_project()
    metadata = dict(project.metadata)
    for key in _ACTIVE_DATASET_METADATA_KEYS:
        metadata.pop(key, None)

    if dataset_name is not None and loaded_dataset is not None:
        metadata.update(
            {
                "active_dataset_name": dataset_name,
                "active_dataset_key": _dataset_identity_key(loaded_dataset),
                "active_dataset_source_locator": loaded_dataset.metadata.source_locator,
                "active_dataset_selected_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    metadata_store.upsert_project(
        ProjectRecord(
            project_id=project.project_id,
            name=project.name,
            metadata=metadata,
            created_at=project.created_at,
            updated_at=datetime.now(timezone.utc),
        )
    )


def _read_persisted_active_dataset_selection(metadata_store) -> tuple[str | None, str | None]:  # noqa: ANN001
    if metadata_store is None:
        return None, None

    project = metadata_store.get_workspace_project()
    metadata = project.metadata if isinstance(project.metadata, dict) else {}
    dataset_name = _normalize_dataset_name(metadata.get("active_dataset_name")) or None
    dataset_key = _normalize_dataset_name(metadata.get("active_dataset_key")) or None
    return dataset_name, dataset_key


def _clear_dataset_results(dataset_name: str | None = None) -> None:
    for key in _WORKFLOW_RESULT_KEYS:
        payload = st.session_state.get(key)
        if not isinstance(payload, dict):
            continue
        if dataset_name is None:
            payload.clear()
            continue
        payload.pop(dataset_name, None)


def _normalize_dataset_name(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()