"""Tests for runtime session state behavior."""

from __future__ import annotations

from app.config.enums import ExecutionBackend, LLMProvider
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
