"""Ollama provider – local model discovery via /api/tags."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config.enums import DEFAULT_MODELS, LLMProvider
from app.providers.base import BaseProvider, ModelItem

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    provider = LLMProvider.OLLAMA

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")

    async def validate_credentials(self) -> bool:
        """Ollama has no auth – just check reachability."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self._base_url}/api/tags", timeout=10)
            return resp.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("Ollama not reachable at %s (%s)", self._base_url, _safe_http_error_detail(exc))
            return False

    async def list_models(self) -> list[ModelItem]:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self._base_url}/api/tags", timeout=15)
                resp.raise_for_status()
            raw = resp.json().get("models", [])
            return self.normalize_model_list(raw)
        except httpx.HTTPError as exc:
            logger.error(
                "Failed to fetch Ollama models from %s (%s)",
                self._base_url,
                _safe_http_error_detail(exc),
            )
            return []

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
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": chosen_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")


def _safe_http_error_detail(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"status={exc.response.status_code}"
    return exc.__class__.__name__
