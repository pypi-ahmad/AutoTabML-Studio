"""Anthropic provider — model discovery and text generation via the official SDK.

Migrated from a hand-rolled ``httpx`` client to the official
``anthropic`` SDK. The SDK handles auth headers (``x-api-key`` and
``anthropic-version``), pagination, retry, and the streaming shape.
"""

from __future__ import annotations

import logging
from typing import Any

from anthropic import AsyncAnthropic

from app.config.enums import DEFAULT_MODELS, LLMProvider
from app.providers.base import BaseProvider, ModelItem

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    provider = LLMProvider.ANTHROPIC

    def __init__(self, api_key: str, *, base_url: str | None = None, timeout: float = 30.0) -> None:
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    # --- credentials ---
    async def validate_credentials(self) -> bool:
        try:
            await self._client.models.list()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Anthropic credential validation failed (%s).", exc.__class__.__name__)
            return False
        return True

    # --- models ---
    async def list_models(self) -> list[ModelItem]:
        try:
            all_models: list[dict[str, Any]] = []
            async for page in self._client.models.list():
                # SDK returns ModelInfo objects; coerce to dict.
                for m in page.data:
                    if hasattr(m, "model_dump"):
                        all_models.append(m.model_dump())
                    elif isinstance(m, dict):
                        all_models.append(m)
                    else:
                        all_models.append({"id": getattr(m, "id", "")})
            return self.normalize_model_list(all_models)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch Anthropic models (%s).", exc.__class__.__name__)
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

    async def generate_text(
        self,
        prompt: str,
        *,
        model_id: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        chosen_model = model_id or self.get_default_model() or "claude-sonnet-4-6"
        message = await self._client.messages.create(
            model=chosen_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
