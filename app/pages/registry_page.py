"""Streamlit page for the model registry."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.security.masking import safe_error_message
from app.state.session import get_or_init_state
from app.tracking.mlflow_query import is_mlflow_available


def render_registry_page() -> None:
    state = get_or_init_state()
    settings = state.settings.tracking
    st.title("🏷️ Model Registry")

    if not is_mlflow_available():
        st.warning("mlflow is not installed. Install with: `pip install mlflow`")
        return

    if not settings.registry_enabled:
        st.info("Model registry is disabled in settings.")
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
        f"'{settings.champion_alias}' and '{settings.candidate_alias}' are aliases. "
        f"'{settings.archived_tag_key}' is the version status tag used for summary display."
    )

    # --- List registered models ---
    try:
        models = service.list_models()
    except RegistryUnavailableError as exc:
        st.warning(
            "Model registry is not available. This MLflow backend does not appear to expose "
            "registry APIs for the current configuration. "
            f"Details: {safe_error_message(exc)}"
        )
        return
    except Exception as exc:
        st.error(f"Failed to query model registry: {safe_error_message(exc)}")
        return

    if not models:
        st.info("No registered models found. Register a model from a previous MLflow artifact source to get started.")
        _render_register_section(service)
        return

    st.subheader("Registered Models")
    model_rows = []
    for model in models:
        model_rows.append({
            "Name": model.name,
            "Versions": model.version_count,
            "Latest": model.latest_version or "N/A",
            "Aliases": ", ".join(f"{k}->v{v}" for k, v in model.aliases.items()) if model.aliases else "",
            "Description": model.description[:80] if model.description else "",
        })
    st.dataframe(pd.DataFrame(model_rows), width="stretch")

    # --- Model detail ---
    model_names = [m.name for m in models]
    selected_model_name = st.selectbox("Inspect model", options=model_names, key="reg_model")

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
                version_rows.append({
                    "Version": version.version,
                    "Status": version.status,
                    "Run ID": (version.run_id or "")[:12],
                    "Aliases": ", ".join(version.aliases) if version.aliases else "",
                    "App Status": version.app_status or "",
                    "Created": version.creation_timestamp,
                })
            st.dataframe(pd.DataFrame(version_rows), width="stretch")

            # --- Promotion actions ---
            st.subheader("Promote Version")
            version_labels = [v.version for v in versions]
            promote_version = st.selectbox(
                "Version to promote", options=version_labels, key="reg_promote_version",
            )
            promote_action = PromotionAction(
                st.selectbox(
                    "Action",
                    options=[a.value for a in PromotionAction],
                    key="reg_promote_action",
                )
            )

            if st.button("Promote", key="reg_promote_button"):
                request = PromotionRequest(
                    model_name=selected_model_name,
                    version=promote_version,
                    action=promote_action,
                )
                try:
                    result = service.promote(request)
                    if result.success:
                        st.success(
                            f"Version {result.version} promoted as {result.action.value}. "
                            + " ".join(result.alias_changes + result.tag_changes)
                        )
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
        "Register a model from an MLflow artifact source. "
        "Copy a full source URI from Run History when available."
    )

    reg_col1, reg_col2 = st.columns(2)
    with reg_col1:
        reg_name = st.text_input("Model name", key="reg_new_name").strip()
    with reg_col2:
        reg_source = st.text_input("Artifact source path/URI", key="reg_new_source").strip()

    reg_run_id = st.text_input("Run ID (optional)", key="reg_new_run_id").strip()
    reg_description = st.text_input("Description (optional)", key="reg_new_description").strip()

    if st.button("Register", key="reg_register_button"):
        if not reg_name or not reg_source:
            st.error("Model name and artifact source are required.")
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
