"""OpenAI provider — model discovery and text generation via the official SDK.

Migrated from a hand-rolled ``httpx`` client to the official
``openai`` SDK. The SDK is the canonical, supported way to talk to
OpenAI, Azure OpenAI, and any OpenAI-compatible endpoint, and brings
official retry/backoff, type-checked responses, and a maintained
``AsyncClient``.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from app.config.enums import DEFAULT_MODELS, LLMProvider
from app.providers.base import BaseProvider, ModelItem

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    provider = LLMProvider.OPENAI

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        organization: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
        )

    # --- credentials ---
    async def validate_credentials(self) -> bool:
        try:
            await self._client.models.list()
        except Exception as exc:  # noqa: BLE001 - any failure means invalid creds
            logger.warning("OpenAI credential validation failed (%s).", exc.__class__.__name__)
            return False
        return True

    # --- models ---
    async def list_models(self) -> list[ModelItem]:
        try:
            response = await self._client.models.list()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch OpenAI models (%s).", exc.__class__.__name__)
            return []
        raw = [{"id": m.id, **getattr(m, "model_dump", dict)()} for m in response.data]
        return self.normalize_model_list(raw)

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

    async def generate_text(
        self,
        prompt: str,
        *,
        model_id: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        chosen_model = model_id or self.get_default_model() or "gpt-4o-mini"
        response = await self._client.chat.completions.create(
            model=chosen_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
