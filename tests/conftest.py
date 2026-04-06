"""Shared test fixtures."""

from __future__ import annotations

import pytest

from app.config.models import AppSettings
from app.state.session import RuntimeState


@pytest.fixture
def default_settings() -> AppSettings:
    return AppSettings()


@pytest.fixture
def runtime_state(default_settings: AppSettings) -> RuntimeState:
    return RuntimeState(settings=default_settings)
