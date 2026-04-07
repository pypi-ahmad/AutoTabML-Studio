"""Base provider interface and common model item shape."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from app.config.enums import LLMProvider


@dataclass
class ModelItem:
    """Normalized internal representation of a model returned by any provider."""
    id: str
    display_name: str
    provider: LLMProvider
    available: bool = True
    is_default: bool = False
    raw_payload: dict[str, Any] = field(default_factory=dict)


class BaseProvider(abc.ABC):
    """Abstract base for all LLM provider integrations."""

    provider: LLMProvider

    @abc.abstractmethod
    async def validate_credentials(self) -> bool:
        """Return True if credentials are valid / reachable."""

    @abc.abstractmethod
    async def list_models(self) -> list[ModelItem]:
        """Fetch models from the provider and return normalized items."""

    @abc.abstractmethod
    def get_default_model(self) -> str | None:
        """Return the fallback default model id for this provider."""

    def normalize_model_list(
        self,
        raw_models: list[dict[str, Any]],
    ) -> list[ModelItem]:
        """Convert raw API response items to the common ModelItem shape.

        Subclasses override to handle provider-specific payloads.
        """
        raise NotImplementedError

    async def generate_text(
        self,
        prompt: str,
        *,
        model_id: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Generate text from the LLM. Override in subclasses.

        Returns the generated text string.  Raises ``NotImplementedError``
        for providers that have not implemented generation yet.
        """
        raise NotImplementedError(
            f"{self.provider.value} provider does not support text generation."
        )
