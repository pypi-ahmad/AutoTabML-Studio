"""Ollama provider — local model discovery and text generation via the official client.

Uses the official ``ollama`` Python SDK which is the maintained
client for the Ollama REST API. The SDK handles retries, async I/O,
and connection pooling; the custom ``httpx`` wrapper is no longer
needed.
"""

from __future__ import annotations

import logging
from typing import Any

from ollama import AsyncClient

from app.config.enums import DEFAULT_MODELS, LLMProvider
from app.providers.base import BaseProvider, ModelItem

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    provider = LLMProvider.OLLAMA

    def __init__(self, base_url: str | None = None, *, timeout: float = 30.0) -> None:
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        # AsyncClient is the official async client; it accepts a
        # custom host and a per-request timeout via the ``timeout``
        # argument on individual calls.
        self._client = AsyncClient(host=self._base_url, timeout=timeout)

    # --- credentials ---
    async def validate_credentials(self) -> bool:
        """Ollama has no auth — just check reachability."""
        try:
            await self._client.list()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama not reachable at %s (%s)", self._base_url, exc.__class__.__name__)
            return False
        return True

    # --- models ---
    async def list_models(self) -> list[ModelItem]:
        try:
            response = await self._client.list()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to fetch Ollama models from %s (%s)",
                self._base_url,
                exc.__class__.__name__,
            )
            return []
        # New SDK returns ListResponse with .models attribute
        models = getattr(response, "models", response) or []
        raw = []
        for m in models:
            if hasattr(m, "model_dump"):
                raw.append(m.model_dump())
            elif isinstance(m, dict):
                raw.append(m)
            else:
                raw.append({"name": getattr(m, "model", "") or getattr(m, "name", "")})
        return self.normalize_model_list(raw)

    def get_default_model(self) -> str | None:
        return DEFAULT_MODELS[LLMProvider.OLLAMA]

    def normalize_model_list(self, raw_models: list[dict[str, Any]]) -> list[ModelItem]:
        default_id = self.get_default_model()
        items: list[ModelItem] = []
        for m in raw_models:
            model_id = m.get("name", "") or m.get("model", "")
            items.append(
                ModelItem(
                    id=model_id,
                    display_name=model_id,
                    provider=LLMProvider.OLLAMA,
                    available=True,
                    is_default=(model_id == default_id) if default_id else False,
                    raw_payload=m,
                )
            )
        return sorted(items, key=lambda x: x.id)

    async def generate_text(
        self,
        prompt: str,
        *,
        model_id: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        chosen_model = model_id or self.get_default_model()
        if not chosen_model:
            models = await self.list_models()
            if not models:
                raise RuntimeError("No Ollama models available. Pull a model first.")
            chosen_model = models[0].id
        response = await self._client.generate(
            model=chosen_model,
            prompt=prompt,
            stream=False,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        )
        return getattr(response, "response", "") or ""
