"""Tests for model token-cost estimates."""

from decimal import Decimal

import pytest

from app.providers.model_pricing import calculate_model_cost, get_model_pricing


@pytest.mark.parametrize(
    ("model_name", "input_price", "output_price"),
    [
        ("Sonnet 5", "2.00", "10.00"),
        ("Gemini Flash 3.7", "0.75", "3.75"),
        ("Gemini 3.5 Flash Lite", "0.30", "2.50"),
        ("GPT 5.6 Luna", "0.20", "1.20"),
        ("GPT 5.6 Terra", "2.00", "12.00"),
        ("Grok 4.6", "2.00", "6.00"),
    ],
)
def test_model_pricing_matches_reference_rates(
    model_name: str,
    input_price: str,
    output_price: str,
) -> None:
    pricing = get_model_pricing(model_name)

    assert pricing.input_price == Decimal(input_price)
    assert pricing.output_price == Decimal(output_price)


def test_calculate_model_cost_prorates_input_and_output_tokens() -> None:
    estimate = calculate_model_cost(
        model_name="GPT 5.6 Luna",
        input_tokens=250_000,
        output_tokens=100_000,
    )

    assert estimate.input_cost == Decimal("0.05")
    assert estimate.output_cost == Decimal("0.12")
    assert estimate.total_cost == Decimal("0.17")
