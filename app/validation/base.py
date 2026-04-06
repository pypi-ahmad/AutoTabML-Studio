"""Base abstraction for validation services."""

from __future__ import annotations

import abc

import pandas as pd

from app.validation.schemas import ValidationResultSummary, ValidationRuleConfig


class BaseValidationService(abc.ABC):
    """Interface that all validation service implementations must satisfy."""

    @abc.abstractmethod
    def validate(
        self,
        df: pd.DataFrame,
        config: ValidationRuleConfig,
        *,
        dataset_name: str | None = None,
    ) -> ValidationResultSummary:
        """Run validation and return a structured summary."""
