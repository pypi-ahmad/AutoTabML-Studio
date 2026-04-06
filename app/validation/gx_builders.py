"""Builders that translate ValidationRuleConfig into GX expectations.

Each builder function returns a list of expectation configuration dicts
suitable for ``BatchDefinition.add_expectation()``.

The builder keeps GX expectation construction isolated so that:
1. We can swap GX API versions without touching rule logic.
2. Not every rule needs to be a GX expectation – some stay in rules.py.
"""

from __future__ import annotations

import logging
from typing import Any

from app.validation.schemas import ValidationRuleConfig

logger = logging.getLogger(__name__)

# Type alias for an expectation config understood by gx_runner
ExpectationSpec = dict[str, Any]


def build_expectations(config: ValidationRuleConfig, column_names: list[str]) -> list[ExpectationSpec]:
    """Build a list of GX expectation specs from the validation config."""
    specs: list[ExpectationSpec] = []

    # NOTE:
    # Row count, required-column presence, and target existence stay in
    # app-level rules for now to avoid duplicate user-facing checks. GX is
    # used here for column-level constraint rules where it adds the most value.

    # Uniqueness checks
    for col in config.uniqueness_columns:
        if col in column_names:
            specs.append({
                "type": "expect_column_values_to_be_unique",
                "kwargs": {"column": col},
            })

    # Numeric range checks
    for col, bounds in config.numeric_range_checks.items():
        if col in column_names:
            kwargs: dict[str, Any] = {"column": col}
            if "min" in bounds:
                kwargs["min_value"] = bounds["min"]
            if "max" in bounds:
                kwargs["max_value"] = bounds["max"]
            specs.append({
                "type": "expect_column_values_to_be_between",
                "kwargs": kwargs,
            })

    # Allowed category checks
    for col, allowed in config.allowed_category_checks.items():
        if col in column_names:
            specs.append({
                "type": "expect_column_values_to_be_in_set",
                "kwargs": {"column": col, "value_set": allowed},
            })

    return specs
