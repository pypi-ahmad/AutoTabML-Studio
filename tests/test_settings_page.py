"""Tests for settings page runtime behaviors that do not require Streamlit rendering."""

from __future__ import annotations

import asyncio

from app.config.enums import LLMProvider
from app.pages.settings_page import _fetch_models
from app.providers.base import ModelItem


class _FakeOllamaProvider:
    async def validate_credentials(self) -> bool:
        return True

    async def list_models(self) -> list[ModelItem]:
        return [
            ModelItem(
                id="llama3:latest",
                display_name="llama3:latest",
                provider=LLMProvider.OLLAMA,
            )
        ]

    def get_default_model(self) -> str | None:
        return None


class TestFetchModels:
    def test_ollama_without_default_does_not_warn(self, runtime_state, monkeypatch):
        runtime_state.provider = LLMProvider.OLLAMA

        monkeypatch.setattr("app.pages.settings_page.build_provider", lambda *args, **kwargs: _FakeOllamaProvider())
        monkeypatch.setattr("app.pages.settings_page._run_async", lambda coro: asyncio.run(coro))

        _fetch_models(runtime_state)

        assert runtime_state.model_fetch_error is None
        assert runtime_state.selected_model_id == "llama3:latest"
