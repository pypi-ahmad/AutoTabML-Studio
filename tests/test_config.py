"""Tests for enums, config defaults, and settings persistence."""

from __future__ import annotations

import json
from pathlib import Path

import tomllib

from app.config.enums import (
    DEFAULT_MODELS,
    PROVIDERS_BY_BACKEND,
    ExecutionBackend,
    LLMProvider,
    WorkspaceMode,
)
from app.config.models import AppSettings, TrackingSettings
from app.config.settings import load_settings, save_settings

# ---------------------------------------------------------------------------
# Enum basics
# ---------------------------------------------------------------------------

class TestEnums:
    def test_workspace_mode_values(self):
        assert WorkspaceMode.DASHBOARD.value == "dashboard"
        assert WorkspaceMode.NOTEBOOK.value == "notebook"

    def test_execution_backend_values(self):
        assert ExecutionBackend.LOCAL.value == "local"
        assert ExecutionBackend.COLAB_MCP.value == "colab_mcp"

    def test_llm_provider_values(self):
        assert set(LLMProvider) == {
            LLMProvider.OPENAI,
            LLMProvider.ANTHROPIC,
            LLMProvider.GEMINI,
            LLMProvider.OLLAMA,
        }


# ---------------------------------------------------------------------------
# Default model mapping
# ---------------------------------------------------------------------------

class TestDefaultModels:
    def test_openai_default(self):
        assert DEFAULT_MODELS[LLMProvider.OPENAI] == "gpt-5.4-mini"

    def test_anthropic_default(self):
        assert DEFAULT_MODELS[LLMProvider.ANTHROPIC] == "claude-sonnet-4-6"

    def test_gemini_default(self):
        assert DEFAULT_MODELS[LLMProvider.GEMINI] == "gemini-2.5-flash"

    def test_ollama_no_default(self):
        assert DEFAULT_MODELS[LLMProvider.OLLAMA] is None


# ---------------------------------------------------------------------------
# Provider filtering by backend
# ---------------------------------------------------------------------------

class TestProvidersByBackend:
    def test_local_includes_ollama(self):
        assert LLMProvider.OLLAMA in PROVIDERS_BY_BACKEND[ExecutionBackend.LOCAL]

    def test_colab_excludes_ollama(self):
        assert LLMProvider.OLLAMA not in PROVIDERS_BY_BACKEND[ExecutionBackend.COLAB_MCP]

    def test_colab_includes_cloud_providers(self):
        colab = PROVIDERS_BY_BACKEND[ExecutionBackend.COLAB_MCP]
        assert LLMProvider.OPENAI in colab
        assert LLMProvider.ANTHROPIC in colab
        assert LLMProvider.GEMINI in colab


# ---------------------------------------------------------------------------
# AppSettings defaults
# ---------------------------------------------------------------------------

class TestAppSettings:
    def test_default_workspace_mode(self, default_settings: AppSettings):
        assert default_settings.workspace_mode == WorkspaceMode.DASHBOARD

    def test_default_execution_backend(self, default_settings: AppSettings):
        assert default_settings.execution.backend == ExecutionBackend.COLAB_MCP

    def test_default_ollama_base_url(self, default_settings: AppSettings):
        assert default_settings.ollama_base_url == "http://localhost:11434"

    def test_default_model_for_provider(self, default_settings: AppSettings):
        assert default_settings.default_model_for_provider(LLMProvider.OPENAI) == "gpt-5.4-mini"
        assert default_settings.default_model_for_provider(LLMProvider.OLLAMA) is None

    def test_pycaret_prefers_gpu_by_default(self, default_settings: AppSettings):
        assert default_settings.pycaret.default_use_gpu is True

    def test_benchmark_prefers_gpu_by_default(self, default_settings: AppSettings):
        assert default_settings.benchmark.prefer_gpu is True

    def test_provider_settings_has_no_api_key_field(self, default_settings: AppSettings):
        assert not hasattr(default_settings.provider, "api_key")

    def test_tracking_alias_maps_to_mlflow(self):
        settings = AppSettings.model_validate(
            {
                "tracking": {
                    "tracking_uri": "sqlite:///mlruns",
                    "registry_enabled": False,
                }
            }
        )

        assert isinstance(settings.tracking, TrackingSettings)
        assert settings.mlflow.tracking_uri == "sqlite:///mlruns"
        assert settings.tracking.registry_enabled is False

    def test_custom_artifact_root_updates_dependent_paths(self, tmp_path: Path):
        root_dir = tmp_path / "runtime"

        settings = AppSettings.model_validate({"artifacts": {"root_dir": str(root_dir)}})

        assert settings.validation.artifacts_dir == root_dir / "validation"
        assert settings.profiling.artifacts_dir == root_dir / "profiling"
        assert settings.benchmark.artifacts_dir == root_dir / "benchmark"
        assert settings.pycaret.artifacts_dir == root_dir / "experiments"
        assert settings.pycaret.models_dir == root_dir / "models"
        assert settings.prediction.artifacts_dir == root_dir / "predictions"
        assert settings.prediction.history_path == root_dir / "predictions" / "history.jsonl"
        assert settings.database.path == root_dir / "app" / "app_metadata.sqlite3"


# ---------------------------------------------------------------------------
# Settings persistence (secrets excluded)
# ---------------------------------------------------------------------------

class TestSettingsPersistence:
    def test_save_contains_no_secret_fields(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("app.config.settings._SETTINGS_DIR", tmp_path)
        monkeypatch.setattr("app.config.settings._SETTINGS_FILE", tmp_path / "settings.json")

        settings = AppSettings()
        save_settings(settings)

        raw = json.loads((tmp_path / "settings.json").read_text())
        # ProviderSettings no longer carries api_key at all
        assert "api_key" not in raw.get("provider", {})

    def test_round_trip(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("app.config.settings._SETTINGS_DIR", tmp_path)
        monkeypatch.setattr("app.config.settings._SETTINGS_FILE", tmp_path / "settings.json")

        settings = AppSettings()
        settings.workspace_mode = WorkspaceMode.NOTEBOOK
        save_settings(settings)

        loaded = load_settings()
        assert loaded.workspace_mode == WorkspaceMode.NOTEBOOK

    def test_load_merges_environment_overrides(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("app.config.settings._SETTINGS_DIR", tmp_path)
        monkeypatch.setattr("app.config.settings._SETTINGS_FILE", tmp_path / "settings.json")

        persisted = {
            "execution": {"backend": "local"},
            "provider": {"base_url": "https://file.example"},
        }
        (tmp_path / "settings.json").write_text(json.dumps(persisted), encoding="utf-8")

        monkeypatch.setenv("AUTOTABML_EXECUTION__BACKEND", "colab_mcp")
        monkeypatch.setenv("AUTOTABML_PROVIDER__BASE_URL", " https://override.example/v1/ ")
        monkeypatch.setenv("AUTOTABML_MLFLOW__TRACKING_URI", "sqlite:///mlruns")

        loaded = load_settings()

        assert loaded.execution.backend == ExecutionBackend.COLAB_MCP
        assert loaded.provider.base_url == "https://override.example/v1/"
        assert loaded.mlflow.tracking_uri == "sqlite:///mlruns"

    def test_load_supports_ollama_base_url_override(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("app.config.settings._SETTINGS_DIR", tmp_path)
        monkeypatch.setattr("app.config.settings._SETTINGS_FILE", tmp_path / "settings.json")
        monkeypatch.setenv("AUTOTABML_OLLAMA_BASE_URL", " http://localhost:11434/ ")

        loaded = load_settings()

        assert loaded.ollama_base_url == "http://localhost:11434"

    def test_save_is_atomic_and_does_not_leave_temp_file(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("app.config.settings._SETTINGS_DIR", tmp_path)
        monkeypatch.setattr("app.config.settings._SETTINGS_FILE", tmp_path / "settings.json")

        save_settings(AppSettings())

        assert (tmp_path / "settings.json").exists()
        assert not (tmp_path / "settings.json.tmp").exists()


class TestPackagingMetadata:
    def test_profiling_extra_pins_compatible_setuptools(self):
        data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        profiling_deps = data["project"]["optional-dependencies"]["profiling"]
        setuptools_dep = next(dep for dep in profiling_deps if dep.startswith("setuptools"))

        assert ">=68" in setuptools_dep
        assert "<82" in setuptools_dep

    def test_experiment_extra_guards_pycaret_on_python_313(self):
        data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        experiment_deps = data["project"]["optional-dependencies"]["experiment"]
        pycaret_dep = next(dep for dep in experiment_deps if dep.startswith("pycaret"))

        assert "python_version < '3.13'" in pycaret_dep

    def test_benchmark_extra_includes_gpu_ready_dependencies(self):
        data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        benchmark_deps = data["project"]["optional-dependencies"]["benchmark"]

        assert any(dep.startswith("lazypredict>=0.3.0") for dep in benchmark_deps)
        assert "xgboost>=2.0" in benchmark_deps
        assert "lightgbm>=4.0" in benchmark_deps
        assert "catboost>=1.2" in benchmark_deps

    def test_experiment_extra_includes_gpu_ready_dependencies(self):
        data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        experiment_deps = data["project"]["optional-dependencies"]["experiment"]

        assert "xgboost>=2.0" in experiment_deps
        assert "lightgbm>=4.0" in experiment_deps
        assert "catboost>=1.2" in experiment_deps
