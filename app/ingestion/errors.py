"""Custom exceptions for dataset ingestion."""

from __future__ import annotations


class IngestionError(Exception):
    """Base exception for user-facing ingestion failures."""


class UnsupportedSourceError(IngestionError):
    """Raised when a source type or locator is unsupported."""


class RemoteAccessError(IngestionError):
    """Raised when a remote resource cannot be reached or inspected safely."""


class EmptyDatasetError(IngestionError):
    """Raised when a loaded dataset has no usable rows or columns."""


class ParseFailureError(IngestionError):
    """Raised when a source is reachable but could not be parsed."""
