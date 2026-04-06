"""Tests for runtime session state behavior."""

from __future__ import annotations

from app.config.enums import PROVIDERS_BY_BACKEND, ExecutionBackend, LLMProvider
from app.providers.base import ModelItem


class TestRuntimeState:
    def test_provider_api_keys_are_scoped_per_provider(self, runtime_state):
        runtime_state.set_provider_api_key(LLMProvider.OPENAI, "openai-secret")
        runtime_state.set_provider_api_key(LLMProvider.ANTHROPIC, "anthropic-secret")

        assert runtime_state.get_provider_api_key(LLMProvider.OPENAI) == "openai-secret"
        assert runtime_state.get_provider_api_key(LLMProvider.ANTHROPIC) == "anthropic-secret"
        assert runtime_state.get_provider_api_key(LLMProvider.GEMINI) is None

    def test_provider_change_clears_model_catalog(self, runtime_state):
        runtime_state.fetched_models = [
            ModelItem(id="gpt-5.4-mini", display_name="GPT", provider=LLMProvider.OPENAI)
        ]
        runtime_state.model_fetch_error = "old error"
        runtime_state.selected_model_id = "gpt-5.4-mini"

        runtime_state.provider = LLMProvider.GEMINI

        assert runtime_state.fetched_models == []
        assert runtime_state.model_fetch_error is None
        assert runtime_state.selected_model_id is None

    def test_backend_change_clears_model_catalog(self, runtime_state):
        runtime_state.fetched_models = [
            ModelItem(id="gpt-5.4-mini", display_name="GPT", provider=LLMProvider.OPENAI)
        ]
        runtime_state.model_fetch_error = "old error"
        runtime_state.selected_model_id = "gpt-5.4-mini"

        # Default is COLAB_MCP, so switch to LOCAL to trigger the change.
        runtime_state.execution_backend = ExecutionBackend.LOCAL

        assert runtime_state.fetched_models == []
        assert runtime_state.model_fetch_error is None
        assert runtime_state.selected_model_id is None

    def test_backend_change_cascades_invalid_provider(self, runtime_state):
        """Switching to colab_mcp while provider=ollama should auto-reset provider."""
        runtime_state.execution_backend = ExecutionBackend.LOCAL
        runtime_state.provider = LLMProvider.OLLAMA

        runtime_state.execution_backend = ExecutionBackend.COLAB_MCP

        allowed = PROVIDERS_BY_BACKEND[ExecutionBackend.COLAB_MCP]
        assert runtime_state.provider in allowed
        assert runtime_state.provider == allowed[0]
        # Model catalog should also be cleared
        assert runtime_state.fetched_models == []
        assert runtime_state.selected_model_id is None

    def test_backend_change_keeps_valid_provider(self, runtime_state):
        """Switching backend when provider is valid for both should keep it."""
        runtime_state.execution_backend = ExecutionBackend.COLAB_MCP
        runtime_state.provider = LLMProvider.OPENAI

        runtime_state.execution_backend = ExecutionBackend.LOCAL

        assert runtime_state.provider == LLMProvider.OPENAI

    def test_same_backend_no_cascade(self, runtime_state):
        """Setting the same backend value should not clear anything."""
        runtime_state.execution_backend = ExecutionBackend.COLAB_MCP
        runtime_state.provider = LLMProvider.OPENAI
        runtime_state.fetched_models = [
            ModelItem(id="gpt-5.4-mini", display_name="GPT", provider=LLMProvider.OPENAI)
        ]
        runtime_state.selected_model_id = "gpt-5.4-mini"

        # Re-set to the same value
        runtime_state.execution_backend = ExecutionBackend.COLAB_MCP

        assert runtime_state.fetched_models != []
        assert runtime_state.selected_model_id == "gpt-5.4-mini"

    def test_to_dict_excludes_backend_valid(self, runtime_state):
        """backend_valid was removed — make sure it's not in to_dict."""
        d = runtime_state.to_dict()
        assert "backend_valid" not in d
