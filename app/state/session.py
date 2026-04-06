"""Centralized session / runtime state management.

Uses Streamlit session_state as the in-memory store but keeps the logic
importable and testable outside Streamlit (for future CLI reuse).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import SecretStr

from app.config.enums import ExecutionBackend, LLMProvider, WorkspaceMode
from app.config.models import AppSettings
from app.config.settings import load_settings, save_settings
from app.providers.base import ModelItem

logger = logging.getLogger(__name__)

_SESSION_KEY = "autotabml_state"


class RuntimeState:
    """Mutable runtime state for the current user session."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings: AppSettings = settings or load_settings()
        self.provider_api_keys: dict[LLMProvider, SecretStr] = {}
        self.fetched_models: list[ModelItem] = []
        self.model_fetch_error: str | None = None
        self.backend_valid: bool | None = None

    # --- convenience accessors ---
    @property
    def workspace_mode(self) -> WorkspaceMode:
        return self.settings.workspace_mode

    @workspace_mode.setter
    def workspace_mode(self, value: WorkspaceMode) -> None:
        self.settings.workspace_mode = value

    @property
    def execution_backend(self) -> ExecutionBackend:
        return self.settings.execution.backend

    @execution_backend.setter
    def execution_backend(self, value: ExecutionBackend) -> None:
        if value != self.settings.execution.backend:
            self.clear_model_catalog()
        self.settings.execution.backend = value

    @property
    def provider(self) -> LLMProvider:
        return self.settings.provider.provider

    @provider.setter
    def provider(self, value: LLMProvider) -> None:
        if value != self.settings.provider.provider:
            self.clear_model_catalog()
        self.settings.provider.provider = value

    @property
    def selected_model_id(self) -> str | None:
        return self.settings.ui.selected_model_id

    @selected_model_id.setter
    def selected_model_id(self, value: str | None) -> None:
        self.settings.ui.selected_model_id = value

    def get_provider_api_key(self, provider: LLMProvider | None = None) -> str | None:
        """Return the session-only API key for the requested provider."""
        selected_provider = provider or self.provider
        secret = self.provider_api_keys.get(selected_provider)
        return secret.get_secret_value() if secret else None

    def set_provider_api_key(self, provider: LLMProvider, value: str) -> None:
        """Store a provider-specific API key in session memory only."""
        cleaned = value.strip()
        if cleaned:
            self.provider_api_keys[provider] = SecretStr(cleaned)

    def clear_model_catalog(self) -> None:
        """Clear fetched models and selection when execution context changes."""
        self.fetched_models = []
        self.model_fetch_error = None
        self.selected_model_id = None

    def persist(self) -> None:
        """Save non-secret settings to disk."""
        save_settings(self.settings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_mode": self.workspace_mode.value,
            "execution_backend": self.execution_backend.value,
            "provider": self.provider.value,
            "selected_model_id": self.selected_model_id,
            "model_count": len(self.fetched_models),
            "backend_valid": self.backend_valid,
        }


# --- Streamlit helpers ---

def get_or_init_state() -> RuntimeState:
    """Retrieve RuntimeState from Streamlit session_state, or create one."""
    try:
        import streamlit as st
        if _SESSION_KEY not in st.session_state:
            st.session_state[_SESSION_KEY] = RuntimeState()
        return st.session_state[_SESSION_KEY]
    except ImportError:
        # Running outside Streamlit (tests, CLI)
        return RuntimeState()
