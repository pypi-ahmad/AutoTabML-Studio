"""Anthropic provider – model discovery via GET /v1/models."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config.enums import DEFAULT_MODELS, LLMProvider
from app.providers.base import BaseProvider, ModelItem

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    provider = LLMProvider.ANTHROPIC

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

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
            logger.warning("Anthropic credential validation failed (%s).", _safe_http_error_detail(exc))
            return False

    async def list_models(self) -> list[ModelItem]:
        try:
            all_models: list[dict[str, Any]] = []
            url: str | None = f"{_BASE_URL}/v1/models"
            async with httpx.AsyncClient() as client:
                while url:
                    resp = await client.get(
                        url,
                        headers=self._auth_headers(),
                        timeout=30,
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    all_models.extend(body.get("data", []))
                    # Anthropic paginates via `has_more` + `last_id`
                    if body.get("has_more"):
                        last_id = body.get("last_id", "")
                        url = f"{_BASE_URL}/v1/models?after_id={last_id}"
                    else:
                        url = None
            return self.normalize_model_list(all_models)
        except httpx.HTTPError as exc:
            logger.error("Failed to fetch Anthropic models (%s).", _safe_http_error_detail(exc))
            return []

    def get_default_model(self) -> str | None:
        return DEFAULT_MODELS[LLMProvider.ANTHROPIC]

    def normalize_model_list(self, raw_models: list[dict[str, Any]]) -> list[ModelItem]:
        default_id = self.get_default_model()
        items: list[ModelItem] = []
        for m in raw_models:
            model_id = m.get("id", "")
            display = m.get("display_name") or model_id
            items.append(
                ModelItem(
                    id=model_id,
                    display_name=display,
                    provider=LLMProvider.ANTHROPIC,
                    available=True,
                    is_default=(model_id == default_id),
                    raw_payload=m,
                )
            )
        return sorted(items, key=lambda x: x.id)

    def _auth_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }


def _safe_http_error_detail(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"status={exc.response.status_code}"
    return exc.__class__.__name__
