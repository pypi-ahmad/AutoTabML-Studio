"""Shared security-layer exceptions."""

from __future__ import annotations


class SecurityError(Exception):
    """Base class for security-boundary failures."""


class TrustedArtifactError(SecurityError):
    """Raised when a persisted local artifact fails trust validation."""