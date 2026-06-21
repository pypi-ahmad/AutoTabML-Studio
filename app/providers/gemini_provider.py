"""Gemini provider — model discovery and text generation via the official SDK.

Migrated from the legacy Google AI ``generativelanguage`` REST endpoint
to the official ``google-genai`` SDK (the new unified Google Gen AI SDK
that supersedes ``google-generativeai``). The new SDK ships typed
responses, official retry, and parity with the Vertex AI surface.
"""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types as genai_types

from app.config.enums import DEFAULT_MODELS, LLMProvider
from app.providers.base import BaseProvider, ModelItem

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    provider = LLMProvider.GEMINI

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        # google-genai takes the API key directly; the http_options
        # allow a custom base URL and timeout (the SDK also accepts
        # a Vertex AI project/location pair — out of scope here).
        http_options = (
            genai_types.HttpOptions(
                base_url=base_url,
                timeout=timeout * 1000,  # google-genai uses milliseconds
            )
            if base_url
            else genai_types.HttpOptions(timeout=int(timeout * 1000))
        )
        self._client = genai.Client(api_key=api_key, http_options=http_options)

    # --- credentials ---
    async def validate_credentials(self) -> bool:
        try:
            await self._client.aio.models.list()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini credential validation failed (%s).", exc.__class__.__name__)
            return False
        return True

    # --- models ---
    async def list_models(self) -> list[ModelItem]:
        try:
            all_raw: list[dict[str, Any]] = []
            async for model in await self._client.aio.models.list():
                payload = model.model_dump() if hasattr(model, "model_dump") else dict(model.__dict__)
                all_raw.append(payload)
            return self.normalize_model_list(all_raw)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch Gemini models (%s).", exc.__class__.__name__)
            return []

    def get_default_model(self) -> str | None:
        return DEFAULT_MODELS[LLMProvider.GEMINI]

    def normalize_model_list(self, raw_models: list[dict[str, Any]]) -> list[ModelItem]:
        default_id = self.get_default_model()
        items: list[ModelItem] = []
        for m in raw_models:
            methods = set(m.get("supportedGenerationMethods", []))
            if methods and not methods & {"generateContent", "streamGenerateContent"}:
                continue  # skip non-text-generation models
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

    async def generate_text(
        self,
        prompt: str,
        *,
        model_id: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        chosen_model = model_id or self.get_default_model() or "gemini-2.5-flash"
        response = await self._client.aio.models.generate_content(
            model=chosen_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return response.text or ""
