"""Catalog service – factory for provider instances and backend-aware filtering."""

from __future__ import annotations

import logging
import os

from app.config.enums import (
    DEFAULT_MODELS,
    PROVIDERS_BY_BACKEND,
    ExecutionBackend,
    LLMProvider,
)
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import BaseProvider, ModelItem
from app.providers.gemini_provider import GeminiProvider
from app.providers.ollama_provider import OllamaProvider
from app.providers.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


def get_allowed_providers(backend: ExecutionBackend) -> list[LLMProvider]:
    """Return providers allowed for the given execution backend."""
    return list(PROVIDERS_BY_BACKEND.get(backend, []))


def build_provider(
    provider: LLMProvider,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseProvider:
    """Instantiate the concrete provider.

    Falls back to environment variables for credentials when *api_key* is not
    supplied explicitly.
    """
    if provider == LLMProvider.OPENAI:
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or provide it in settings.")
        return OpenAIProvider(api_key=key)

    if provider == LLMProvider.ANTHROPIC:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY or provide it in settings.")
        return AnthropicProvider(api_key=key)

    if provider == LLMProvider.GEMINI:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY or provide it in settings.")
        return GeminiProvider(api_key=key)

    if provider == LLMProvider.OLLAMA:
        url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaProvider(base_url=url)

    raise ValueError(f"Unknown provider: {provider}")


def resolve_default_model(
    models: list[ModelItem],
    provider: LLMProvider,
) -> ModelItem | None:
    """Pick the best default from a fetched model list.

    1. If the hardcoded default exists in the list, return it.
    2. Otherwise return the first available model and log a warning.
    3. If the list is empty, return None.
    """
    if not models:
        return None

    default_id = DEFAULT_MODELS.get(provider)
    if default_id:
        for m in models:
            if m.id == default_id:
                return m
        logger.warning(
            "Default model '%s' not found in %s model list. "
            "Falling back to first available model.",
            default_id,
            provider.value,
        )

    return models[0] if models else None
