"""Tests for provider abstractions, catalog service, and model normalization."""

from __future__ import annotations

import pytest

from app.config.enums import DEFAULT_MODELS, ExecutionBackend, LLMProvider
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import ModelItem
from app.providers.catalog_service import (
    build_provider,
    get_allowed_providers,
    resolve_default_model,
)
from app.providers.gemini_provider import GeminiProvider
from app.providers.ollama_provider import OllamaProvider
from app.providers.openai_provider import OpenAIProvider

# ---------------------------------------------------------------------------
# get_allowed_providers
# ---------------------------------------------------------------------------

class TestAllowedProviders:
    def test_local_returns_all_four(self):
        result = get_allowed_providers(ExecutionBackend.LOCAL)
        assert len(result) == 4
        assert LLMProvider.OLLAMA in result

    def test_colab_excludes_ollama(self):
        result = get_allowed_providers(ExecutionBackend.COLAB_MCP)
        assert LLMProvider.OLLAMA not in result
        assert len(result) == 3


# ---------------------------------------------------------------------------
# build_provider
# ---------------------------------------------------------------------------

class TestBuildProvider:
    def test_openai_requires_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            build_provider(LLMProvider.OPENAI, api_key="")

    def test_anthropic_requires_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            build_provider(LLMProvider.ANTHROPIC, api_key="")

    def test_gemini_requires_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            build_provider(LLMProvider.GEMINI, api_key="")

    def test_openai_returns_correct_type(self):
        p = build_provider(LLMProvider.OPENAI, api_key="test-key")
        assert isinstance(p, OpenAIProvider)

    def test_anthropic_returns_correct_type(self):
        p = build_provider(LLMProvider.ANTHROPIC, api_key="test-key")
        assert isinstance(p, AnthropicProvider)

    def test_gemini_returns_correct_type(self):
        p = build_provider(LLMProvider.GEMINI, api_key="test-key")
        assert isinstance(p, GeminiProvider)

    def test_ollama_no_key_required(self):
        p = build_provider(LLMProvider.OLLAMA)
        assert isinstance(p, OllamaProvider)


# ---------------------------------------------------------------------------
# resolve_default_model
# ---------------------------------------------------------------------------

def _make_items(ids: list[str], provider: LLMProvider) -> list[ModelItem]:
    default_id = DEFAULT_MODELS.get(provider)
    return [
        ModelItem(
            id=mid,
            display_name=mid,
            provider=provider,
            is_default=(mid == default_id),
        )
        for mid in ids
    ]


class TestResolveDefaultModel:
    def test_returns_default_when_present(self):
        items = _make_items(["gpt-4", "gpt-5.4-mini", "gpt-6"], LLMProvider.OPENAI)
        result = resolve_default_model(items, LLMProvider.OPENAI)
        assert result is not None
        assert result.id == "gpt-5.4-mini"

    def test_falls_back_to_first_when_default_absent(self):
        items = _make_items(["gpt-4", "gpt-6"], LLMProvider.OPENAI)
        result = resolve_default_model(items, LLMProvider.OPENAI)
        assert result is not None
        assert result.id == "gpt-4"

    def test_returns_none_for_empty_list(self):
        result = resolve_default_model([], LLMProvider.OPENAI)
        assert result is None

    def test_ollama_with_no_default_uses_first(self):
        items = _make_items(["llama3", "codellama"], LLMProvider.OLLAMA)
        result = resolve_default_model(items, LLMProvider.OLLAMA)
        assert result is not None
        assert result.id == "llama3"


# ---------------------------------------------------------------------------
# Model normalization
# ---------------------------------------------------------------------------

class TestModelNormalization:
    def test_openai_normalization(self):
        provider = OpenAIProvider(api_key="k")
        raw = [{"id": "gpt-5.4-mini", "owned_by": "openai"}]
        items = provider.normalize_model_list(raw)
        assert len(items) == 1
        assert items[0].id == "gpt-5.4-mini"
        assert items[0].provider == LLMProvider.OPENAI
        assert items[0].is_default is True

    def test_anthropic_normalization(self):
        provider = AnthropicProvider(api_key="k")
        raw = [{"id": "claude-sonnet-4-6", "display_name": "Claude Sonnet 4.6"}]
        items = provider.normalize_model_list(raw)
        assert len(items) == 1
        assert items[0].display_name == "Claude Sonnet 4.6"
        assert items[0].is_default is True

    def test_gemini_filters_non_text_models(self):
        provider = GeminiProvider(api_key="k")
        raw = [
            {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash",
             "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embedding-001", "displayName": "Embedding",
             "supportedGenerationMethods": ["embedContent"]},
        ]
        items = provider.normalize_model_list(raw)
        assert len(items) == 1
        assert items[0].id == "gemini-2.5-flash"
        assert items[0].is_default is True

    def test_gemini_strips_models_prefix(self):
        provider = GeminiProvider(api_key="k")
        raw = [
            {"name": "models/gemini-pro", "supportedGenerationMethods": ["generateContent"]},
        ]
        items = provider.normalize_model_list(raw)
        assert items[0].id == "gemini-pro"

    def test_gemini_uses_header_auth(self):
        provider = GeminiProvider(api_key="secret-key")
        assert provider._auth_headers() == {"x-goog-api-key": "secret-key"}

    def test_ollama_normalization(self):
        provider = OllamaProvider()
        raw = [{"name": "llama3:latest"}, {"name": "codellama:7b"}]
        items = provider.normalize_model_list(raw)
        assert len(items) == 2
        assert items[0].provider == LLMProvider.OLLAMA

    def test_openai_sorted_by_id(self):
        provider = OpenAIProvider(api_key="k")
        raw = [{"id": "z-model"}, {"id": "a-model"}]
        items = provider.normalize_model_list(raw)
        assert items[0].id == "a-model"
        assert items[1].id == "z-model"
