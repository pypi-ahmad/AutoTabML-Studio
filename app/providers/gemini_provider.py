"""Gemini provider – model discovery via the Google AI generativelanguage API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config.enums import DEFAULT_MODELS, LLMProvider
from app.providers.base import BaseProvider, ModelItem

logger = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Only show models whose supportedGenerationMethods include text generation
_TEXT_GEN_METHODS = {"generateContent", "streamGenerateContent"}


class GeminiProvider(BaseProvider):
    provider = LLMProvider.GEMINI

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def validate_credentials(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{_BASE_URL}/models",
                    headers=self._auth_headers(),
                    timeout=15,
                )
            return resp.status_code == 200
        except httpx.HTTPError as exc:
            logger.warning("Gemini credential validation failed (%s).", _safe_http_error_detail(exc))
            return False

    async def list_models(self) -> list[ModelItem]:
        try:
            all_raw: list[dict[str, Any]] = []
            page_token: str | None = None
            async with httpx.AsyncClient() as client:
                while True:
                    params: dict[str, str] = {}
                    if page_token:
                        params["pageToken"] = page_token
                    resp = await client.get(
                        f"{_BASE_URL}/models",
                        headers=self._auth_headers(),
                        params=params,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    all_raw.extend(body.get("models", []))
                    page_token = body.get("nextPageToken")
                    if not page_token:
                        break
            return self.normalize_model_list(all_raw)
        except httpx.HTTPError as exc:
            logger.error("Failed to fetch Gemini models (%s).", _safe_http_error_detail(exc))
            return []

    def get_default_model(self) -> str | None:
        return DEFAULT_MODELS[LLMProvider.GEMINI]

    def normalize_model_list(self, raw_models: list[dict[str, Any]]) -> list[ModelItem]:
        default_id = self.get_default_model()
        items: list[ModelItem] = []
        for m in raw_models:
            methods = set(m.get("supportedGenerationMethods", []))
            if not methods & _TEXT_GEN_METHODS:
                continue  # skip non-text-generation models
            # name comes as "models/gemini-..." – strip the prefix for a cleaner id
            full_name: str = m.get("name", "")
            model_id = full_name.removeprefix("models/")
            display = m.get("displayName") or model_id
            items.append(
                ModelItem(
                    id=model_id,
                    display_name=display,
                    provider=LLMProvider.GEMINI,
                    available=True,
                    is_default=(model_id == default_id),
                    raw_payload=m,
                )
            )
        return sorted(items, key=lambda x: x.id)

    def _auth_headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key}


def _safe_http_error_detail(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"status={exc.response.status_code}"
    return exc.__class__.__name__
