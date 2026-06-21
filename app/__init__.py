"""Core package metadata for AutoTabML Studio."""

from __future__ import annotations

APP_NAME = "AutoTabML Studio"
DIST_NAME = "autotabml-studio"
CLI_ENTRYPOINT = "autotabml"
STREAMLIT_ENTRYPOINT = "app/main.py"
__version__ = "0.2.0"

__all__ = [
    "APP_NAME",
    "CLI_ENTRYPOINT",
    "DIST_NAME",
    "STREAMLIT_ENTRYPOINT",
    "__version__",
]
