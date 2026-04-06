"""OpenAI provider – model discovery via GET /v1/models."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config.enums import DEFAULT_MODELS, LLMProvider
from app.providers.base import BaseProvider, ModelItem

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.openai.com"


class OpenAIProvider(BaseProvider):
    provider = LLMProvider.OPENAI

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    # --- credentials ---
    async def validate_credentials(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{_BASE_URL}/v1/models",
                    headers=self._auth_headers(),
                    timeout=15,
                )
            return resp.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("OpenAI credential validation failed (%s).", _safe_http_error_detail(exc))
            return False

    # --- models ---
    async def list_models(self) -> list[ModelItem]:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{_BASE_URL}/v1/models",
                    headers=self._auth_headers(),
                    timeout=30,
                )
                resp.raise_for_status()
            raw = resp.json().get("data", [])
            return self.normalize_model_list(raw)
        except httpx.HTTPError as exc:
            logger.error("Failed to fetch OpenAI models (%s).", _safe_http_error_detail(exc))
            return []

    def get_default_model(self) -> str | None:
        return DEFAULT_MODELS[LLMProvider.OPENAI]

    def normalize_model_list(self, raw_models: list[dict[str, Any]]) -> list[ModelItem]:
        default_id = self.get_default_model()
        items: list[ModelItem] = []
        for m in raw_models:
            model_id = m.get("id", "")
            items.append(
                ModelItem(
                    id=model_id,
                    display_name=model_id,
                    provider=LLMProvider.OPENAI,
                    available=True,
                    is_default=(model_id == default_id),
                    raw_payload=m,
                )
            )
        return sorted(items, key=lambda x: x.id)

    # --- internals ---
    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}


def _safe_http_error_detail(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"status={exc.response.status_code}"
    return exc.__class__.__name__
