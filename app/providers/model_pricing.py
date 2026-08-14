"""Reference prices and token-cost estimates for hosted AI models."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

TOKENS_PER_MILLION = Decimal("1000000")


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Standard per-million-token prices for a hosted model."""

    name: str
    input_price: Decimal
    output_price: Decimal
    details: str


@dataclass(frozen=True, slots=True)
class ModelCostEstimate:
    """Estimated input, output, and combined cost in US dollars."""

    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal


MODEL_PRICING = (
    ModelPricing(
        name="Sonnet 5",
        input_price=Decimal("2.00"),
        output_price=Decimal("10.00"),
        details="Standard $2/$10 pricing is permanent. Batch API requests receive a 50% discount.",
    ),
    ModelPricing(
        name="Gemini Flash 3.7",
        input_price=Decimal("0.75"),
        output_price=Decimal("3.75"),
        details="Promo pricing is half the 3.6 Flash rate through December 31, 2026.",
    ),
    ModelPricing(
        name="Gemini 3.5 Flash Lite",
        input_price=Decimal("0.30"),
        output_price=Decimal("2.50"),
        details="Google's lowest-cost Flash tier. Batch pricing is $0.15 input and $1.25 output.",
    ),
    ModelPricing(
        name="GPT 5.6 Luna",
        input_price=Decimal("0.20"),
        output_price=Decimal("1.20"),
        details="OpenAI's budget tier. Prompt cache reads cost $0.02 per 1M tokens.",
    ),
    ModelPricing(
        name="GPT 5.6 Terra",
        input_price=Decimal("2.00"),
        output_price=Decimal("12.00"),
        details="OpenAI's mid-tier model. Pricing doubles above 272k input tokens.",
    ),
    ModelPricing(
        name="Grok 4.6",
        input_price=Decimal("2.00"),
        output_price=Decimal("6.00"),
        details="Fast mode or prompts above 200k tokens use the higher $4 input / $12 output rate.",
    ),
)


def get_model_pricing(model_name: str) -> ModelPricing:
    """Return pricing for a supported model name."""
    for pricing in MODEL_PRICING:
        if pricing.name == model_name:
            return pricing
    raise ValueError(f"Unknown model pricing: {model_name}")


def calculate_model_cost(model_name: str, input_tokens: int, output_tokens: int) -> ModelCostEstimate:
    """Estimate standard token costs for a supported model."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative")

    pricing = get_model_pricing(model_name)
    input_cost = Decimal(input_tokens) * pricing.input_price / TOKENS_PER_MILLION
    output_cost = Decimal(output_tokens) * pricing.output_price / TOKENS_PER_MILLION
    return ModelCostEstimate(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
    )
