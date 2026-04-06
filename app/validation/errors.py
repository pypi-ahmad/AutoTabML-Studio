"""Custom exceptions for the validation layer."""

from __future__ import annotations


class ValidationError(Exception):
    """Base exception for validation failures."""


class ValidationSetupError(ValidationError):
    """Raised when the validation infrastructure cannot be initialized."""


class RuleConfigError(ValidationError):
    """Raised when a validation rule configuration is invalid."""
