"""Settings persistence – load / save runtime settings to a local JSON file.

Secrets are NEVER written to the settings file.  They come exclusively from
environment variables or the in-memory session.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.models import (
    AppSettings,
    ArtifactSettings,
    BenchmarkSettings,
    DatabaseSettings,
    ExecutionSettings,
    MLflowSettings,
    PredictionSettings,
    ProfilingSettings,
    ProviderSettings,
    PyCaretExperimentSettings,
    UISettings,
    ValidationSettings,
)

logger = logging.getLogger(__name__)

_SETTINGS_DIR = Path.home() / ".autotabml"
_SETTINGS_FILE = _SETTINGS_DIR / "settings.json"




class _EnvironmentSettings(BaseSettings):
    """Optional environment overrides for local runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="AUTOTABML_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    workspace_mode: str | None = None
    execution: ExecutionSettings | None = None
    provider: ProviderSettings | None = None
    ui: UISettings | None = None
    artifacts: ArtifactSettings | None = None
    database: DatabaseSettings | None = None
    validation: ValidationSettings | None = None
    profiling: ProfilingSettings | None = None
    benchmark: BenchmarkSettings | None = None
    pycaret: PyCaretExperimentSettings | None = None
    mlflow: MLflowSettings | None = None
    prediction: PredictionSettings | None = None
    ollama_base_url: str | None = None


def load_settings() -> AppSettings:
    """Load settings from the local JSON file, falling back to defaults."""
    load_dotenv(override=False)

    raw: dict = {}
    if _SETTINGS_FILE.exists():
        try:
            raw = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Corrupted settings file - using defaults: %s", exc)
            raw = {}

    try:
        env_overrides = _EnvironmentSettings().model_dump(exclude_none=True, mode="json")
    except Exception as exc:  # pragma: no cover - depends on environment misuse
        raise ValueError(f"Invalid AUTOTABML_* environment configuration: {exc}") from exc

    merged = _deep_merge(raw, env_overrides)
    return AppSettings.model_validate(merged)


def save_settings(settings: AppSettings) -> None:
    """Persist settings to ~/.autotabml/settings.json (secrets excluded)."""
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    data = settings.model_dump(mode="json")
    temp_path = _SETTINGS_FILE.with_name(f"{_SETTINGS_FILE.name}.tmp")
    temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp_path.replace(_SETTINGS_FILE)
    logger.info("Settings saved to %s", _SETTINGS_FILE)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
