"""Streamlit page for the model registry."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.pages.shared_history import render_past_runs_section, render_saved_models_section
from app.pages.dataset_workspace import go_to_page
from app.pages.ui_labels import format_enum_value, PROMOTION_LABELS, make_format_func
from app.security.masking import safe_error_message
from app.state.session import get_or_init_state
from app.storage import build_metadata_store
from app.storage.models import AppJobType
from app.tracking.mlflow_query import is_mlflow_available


def render_registry_page() -> None:
    state = get_or_init_state()
    settings = state.settings.tracking
    metadata_store = build_metadata_store(state.settings)
    st.title("🏷️ Model Registry")
    st.caption(
        "The registry keeps versioned copies of your best models. "
        "Promote a version to 'Champion' when it's ready for production use."
    )

    # ── Past runs & saved models (always visible) ─────────────────────
    render_past_runs_section(metadata_store, AppJobType.EXPERIMENT, key_prefix="reg")
    render_saved_models_section(state.settings.prediction, key_prefix="reg")

    st.divider()

    if not is_mlflow_available():
        st.warning(
            "The model registry requires experiment tracking to be set up. "
            "Please ask your administrator to configure it."
        )
        return

    if not settings.registry_enabled:
        st.info("The model registry is turned off.")
        if st.button("Go to Settings", key="reg_goto_settings"):
            go_to_page("Settings")
        return

    from app.registry.errors import RegistryUnavailableError
    from app.registry.registry_service import RegistryService
    from app.registry.schemas import PromotionAction, PromotionRequest

    service = RegistryService(
        tracking_uri=settings.tracking_uri,
        registry_uri=settings.registry_uri,
        champion_alias=settings.champion_alias,
        candidate_alias=settings.candidate_alias,
        archived_tag_key=settings.archived_tag_key,
    )
    st.caption(
        f"Models can be promoted to **{settings.champion_alias}** (production-ready) "
        f"or **{settings.candidate_alias}** (testing). "
        "Archived versions are tagged automatically."
    )

    # --- List registered models ---
    try:
        models = service.list_models()
    except RegistryUnavailableError as exc:
        st.warning(
            "Model registry is not available for the current configuration. "
            "Check your tracking settings or ask your administrator."
        )
        with st.expander("Technical details", expanded=False):
            st.caption(f"Error: {safe_error_message(exc)}")
        return
    except Exception as exc:
        st.error(f"Failed to query model registry: {safe_error_message(exc)}")
        return

    if not models:
        st.info("No registered models found. Register a model from a previous training run to get started.")
        _render_register_section(service)
        return

    st.subheader("Registered Models")

    # ── Executive summary ──────────────────────────────────────────────
    _total_versions = sum(m.version_count for m in models)
    _champion_count = sum(1 for m in models if any(k.lower() == settings.champion_alias.lower() for k in m.aliases))
    _candidate_count = sum(1 for m in models if any(k.lower() == settings.candidate_alias.lower() for k in m.aliases))
    st.info(
        f"**Overview:** **{len(models)}** registered model(s) with **{_total_versions}** total version(s). "
        + (f"**{_champion_count}** champion(s), **{_candidate_count}** candidate(s)." if _champion_count or _candidate_count else "No champions or candidates promoted yet.")
    )

    model_rows = []
    for model in models:
        model_rows.append({
            "Name": model.name,
            "Versions": model.version_count,
            "Latest": model.latest_version or "N/A",
            "Version labels": ", ".join(f"{k}->v{v}" for k, v in model.aliases.items()) if model.aliases else "",
            "Description": model.description[:80] if model.description else "",
        })
    st.dataframe(pd.DataFrame(model_rows), width="stretch")

    # --- Model detail ---
    model_names = [m.name for m in models]
    selected_model_name = st.selectbox("Inspect model", options=model_names, key="reg_model", help="Pick a registered model to see its versions and details.")

    if selected_model_name:
        try:
            versions = service.list_versions(selected_model_name)
        except Exception as exc:
            st.error(f"Failed to list versions: {safe_error_message(exc)}")
            return

        if versions:
            st.subheader(f"Versions of `{selected_model_name}`")
            version_rows = []
            for version in versions:
                # Readiness badge from aliases/status
                if version.aliases:
                    _alias_lower = {a.lower() for a in version.aliases}
                    if "champion" in _alias_lower:
                        _readiness = "⭐ Champion"
                    elif "candidate" in _alias_lower:
                        _readiness = "🧪 Candidate"
                    else:
                        _readiness = ", ".join(version.aliases)
                elif version.app_status:
                    _readiness = format_enum_value(version.app_status)
                else:
                    _readiness = "—"
                version_rows.append({
                    "Version": version.version,
                    "Readiness": _readiness,
                    "Version labels": ", ".join(version.aliases) if version.aliases else "",
                    "Created": version.creation_timestamp,
                })
            st.dataframe(pd.DataFrame(version_rows), width="stretch")

            # Show run IDs for traceability (power-user need)
            _versions_with_run_id = [v for v in versions if v.run_id]
            if _versions_with_run_id:
                with st.expander("Technical details", expanded=False):
                    for v in _versions_with_run_id:
                        st.caption(f"Version {v.version} — tracking run: `{v.run_id[:12]}…` | status: {v.status or '—'}")

            # --- Promotion / lifecycle actions ---
            st.subheader("Manage Version")
            version_labels = [v.version for v in versions]
            promote_version = st.selectbox(
                "Version to promote", options=version_labels, key="reg_promote_version",
                help="Which version to change the status of.",
            )
            promote_action = PromotionAction(
                st.selectbox(
                    "Action",
                    options=[a.value for a in PromotionAction],
                    format_func=make_format_func(PROMOTION_LABELS),
                    key="reg_promote_action",
                    help="Choose what to do with this version.",
                )
            )

            if st.button("Apply", key="reg_promote_button"):
                request = PromotionRequest(
                    model_name=selected_model_name,
                    version=promote_version,
                    action=promote_action,
                )
                try:
                    result = service.promote(request)
                    if result.success:
                        _action_label = PROMOTION_LABELS.get(result.action.value, format_enum_value(result.action.value))
                        st.success(
                            f"Version {result.version}: {_action_label}."
                        )
                        if result.alias_changes or result.tag_changes:
                            with st.expander("Details", expanded=False):
                                for change in result.alias_changes + result.tag_changes:
                                    st.caption(change)
                    for warning in result.warnings:
                        st.warning(warning)
                except Exception as exc:
                    st.error(f"Promotion failed: {safe_error_message(exc)}")
        else:
            st.info(f"No versions found for model `{selected_model_name}`.")

    _render_register_section(service)


def _render_register_section(service) -> None:
    """Render the manual model registration form."""

    st.subheader("Register Model")
    st.caption(
        "Add a model to the registry so you can version, promote, and track it. "
        "You can find the source path in **History**."
    )

    reg_col1, reg_col2 = st.columns(2)
    with reg_col1:
        reg_name = st.text_input("Model name", key="reg_new_name", help="A short, descriptive name for this model (e.g. 'sales-forecast-v2').").strip()
    with reg_col2:
        reg_source = st.text_input("Source path", key="reg_new_source", help="Path or URI to the model file. Copy this from Run History.").strip()

    reg_run_id = st.text_input("Tracking run ID (optional)", key="reg_new_run_id", help="The experiment tracking run that produced this model. Leave blank if not applicable.").strip()
    reg_description = st.text_input("Description (optional)", key="reg_new_description", help="A brief note about this model — e.g. what data it was trained on or its intended use.").strip()

    if st.button("Register", key="reg_register_button"):
        if not reg_name or not reg_source:
            st.error("Model name and source path are required.")
            return
        try:
            version = service.register_model(
                reg_name,
                source=reg_source,
                run_id=reg_run_id or None,
                description=reg_description,
            )
            st.success(
                f"Registered model '{version.model_name}' version {version.version}."
            )
        except Exception as exc:
            st.error(f"Registration failed: {safe_error_message(exc)}")
