"""Tests for the model registry layer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config.models import TrackingSettings
from app.registry.errors import ModelNotFoundError, PromotionError, RegistryUnavailableError
from app.registry.registry_service import RegistryService
from app.registry.schemas import (
    PromotionAction,
    PromotionRequest,
    PromotionResult,
    RegistryModelSummary,
    RegistryVersionSummary,
)


# ---------------------------------------------------------------------------
# Fake MLflow objects
# ---------------------------------------------------------------------------


class _FakeRegisteredModel:
    def __init__(self, name, description="", tags=None, aliases=None, latest_versions=None):
        self.name = name
        self.description = description
        self.creation_timestamp = 1704067200000
        self.last_updated_timestamp = 1704067200000
        self.tags = tags or {}
        self.aliases = aliases or {}
        self.latest_versions = latest_versions or []


class _FakeModelVersion:
    def __init__(self, name, version, run_id=None, source=None, status="READY", tags=None, aliases=None):
        self.name = name
        self.version = version
        self.creation_timestamp = 1704067200000
        self.last_updated_timestamp = 1704067200000
        self.description = ""
        self.source = source or f"file:///models/{name}/{version}"
        self.run_id = run_id
        self.run_link = ""
        self.status = status
        self.tags = tags or {}
        self.aliases = aliases or []


def _patch_registry(monkeypatch, models=None, versions=None):
    """Patch mlflow_query functions used by RegistryService."""

    from app.tracking import mlflow_query

    all_models = list(models or [])
    all_versions = list(versions or [])
    created_models = []
    created_versions = []
    alias_log = []
    tag_log = []
    version_tag_log = []
    deleted_version_tags = []
    deleted_aliases = []

    monkeypatch.setattr(mlflow_query, "is_mlflow_available", lambda: True)
    monkeypatch.setattr(mlflow_query, "_require_mlflow", lambda: None)

    def fake_list_registered_models(*, tracking_uri=None, registry_uri=None):
        from app.tracking.mlflow_query import _normalize_registered_model
        return [_normalize_registered_model(m) for m in all_models + created_models]

    def fake_get_registered_model(name, *, tracking_uri=None, registry_uri=None):
        from app.tracking.mlflow_query import _normalize_registered_model
        for m in all_models + created_models:
            if m.name == name:
                return _normalize_registered_model(m)
        raise ModelNotFoundError(f"Model '{name}' not found.")

    def fake_list_model_versions(name, *, tracking_uri=None, registry_uri=None):
        from app.tracking.mlflow_query import _normalize_model_version
        return [_normalize_model_version(v) for v in all_versions + created_versions if v.name == name]

    def fake_get_model_version(name, version, *, tracking_uri=None, registry_uri=None):
        from app.tracking.mlflow_query import _normalize_model_version
        for v in all_versions + created_versions:
            if v.name == name and str(v.version) == str(version):
                return _normalize_model_version(v)
        from app.registry.errors import VersionNotFoundError
        raise VersionNotFoundError(f"Version '{version}' of model '{name}' not found.")

    def fake_create_registered_model(name, *, description="", tags=None, tracking_uri=None, registry_uri=None):
        from app.tracking.mlflow_query import _normalize_registered_model
        model = _FakeRegisteredModel(name, description=description, tags=tags or {})
        created_models.append(model)
        return _normalize_registered_model(model)

    def fake_create_model_version(name, *, source, run_id=None, description="", tags=None, tracking_uri=None, registry_uri=None):
        from app.tracking.mlflow_query import _normalize_model_version
        version = _FakeModelVersion(
            name,
            str(len([v for v in all_versions + created_versions if v.name == name]) + 1),
            run_id=run_id,
            source=source,
            tags=tags or {},
        )
        created_versions.append(version)
        return _normalize_model_version(version)

    def fake_set_model_alias(name, alias, version, *, tracking_uri=None, registry_uri=None):
        alias_log.append((name, alias, version))
        for model in all_models + created_models:
            if model.name == name:
                model.aliases = dict(model.aliases)
                model.aliases[alias] = version

    def fake_delete_model_alias(name, alias, *, tracking_uri=None, registry_uri=None):
        deleted_aliases.append((name, alias))
        for model in all_models + created_models:
            if model.name == name and alias in model.aliases:
                model.aliases = dict(model.aliases)
                model.aliases.pop(alias, None)

    def fake_set_model_tag(name, key, value, *, tracking_uri=None, registry_uri=None):
        tag_log.append((name, key, value))

    def fake_set_model_version_tag(name, version, key, value, *, tracking_uri=None, registry_uri=None):
        version_tag_log.append((name, version, key, value))
        for model_version in all_versions + created_versions:
            if model_version.name == name and str(model_version.version) == str(version):
                model_version.tags[key] = value

    def fake_delete_model_version_tag(name, version, key, *, tracking_uri=None, registry_uri=None):
        deleted_version_tags.append((name, version, key))
        for model_version in all_versions + created_versions:
            if model_version.name == name and str(model_version.version) == str(version):
                model_version.tags.pop(key, None)

    monkeypatch.setattr(mlflow_query, "list_registered_models", fake_list_registered_models)
    monkeypatch.setattr(mlflow_query, "get_registered_model", fake_get_registered_model)
    monkeypatch.setattr(mlflow_query, "list_model_versions", fake_list_model_versions)
    monkeypatch.setattr(mlflow_query, "get_model_version", fake_get_model_version)
    monkeypatch.setattr(mlflow_query, "create_registered_model", fake_create_registered_model)
    monkeypatch.setattr(mlflow_query, "create_model_version", fake_create_model_version)
    monkeypatch.setattr(mlflow_query, "set_model_alias", fake_set_model_alias)
    monkeypatch.setattr(mlflow_query, "delete_model_alias", fake_delete_model_alias)
    monkeypatch.setattr(mlflow_query, "set_model_tag", fake_set_model_tag)
    monkeypatch.setattr(mlflow_query, "set_model_version_tag", fake_set_model_version_tag)
    monkeypatch.setattr(mlflow_query, "delete_model_version_tag", fake_delete_model_version_tag)

    return SimpleNamespace(
        created_models=created_models,
        created_versions=created_versions,
        alias_log=alias_log,
        tag_log=tag_log,
        version_tag_log=version_tag_log,
        deleted_version_tags=deleted_version_tags,
        deleted_aliases=deleted_aliases,
    )


# ---------------------------------------------------------------------------
# Registry service
# ---------------------------------------------------------------------------


class TestRegistryServiceListModels:
    def test_list_models_returns_summaries(self, monkeypatch):
        _patch_registry(monkeypatch, models=[
            _FakeRegisteredModel("model-a"),
            _FakeRegisteredModel("model-b"),
        ])

        service = RegistryService()
        models = service.list_models()

        assert len(models) == 2
        assert models[0].name == "model-a"
        assert models[1].name == "model-b"

    def test_list_models_empty(self, monkeypatch):
        _patch_registry(monkeypatch)

        service = RegistryService()
        models = service.list_models()

        assert models == []

    def test_list_models_uses_actual_version_count(self, monkeypatch):
        from app.tracking import mlflow_query

        model = _FakeRegisteredModel(
            "model-a",
            latest_versions=[_FakeModelVersion("model-a", "2")],
        )
        versions = [
            _FakeModelVersion("model-a", "1"),
            _FakeModelVersion("model-a", "2"),
            _FakeModelVersion("model-a", "3"),
        ]

        monkeypatch.setattr(mlflow_query, "is_mlflow_available", lambda: True)
        monkeypatch.setattr(mlflow_query, "_require_mlflow", lambda: None)
        monkeypatch.setattr(
            mlflow_query,
            "_get_client",
            lambda tracking_uri=None, registry_uri=None: SimpleNamespace(
                search_registered_models=lambda: [model],
                search_model_versions=lambda filter_string: versions,
            ),
        )

        models = RegistryService().list_models()

        assert models[0].version_count == 3
        assert models[0].latest_version == "3"

    def test_list_models_coerces_integer_alias_versions_to_strings(self, monkeypatch):
        _patch_registry(monkeypatch, models=[
            _FakeRegisteredModel("model-a", aliases={"champion": 1}),
        ])

        models = RegistryService().list_models()

        assert models[0].aliases == {"champion": "1"}


class TestRegistryServiceVersions:
    def test_list_versions(self, monkeypatch):
        _patch_registry(monkeypatch, versions=[
            _FakeModelVersion("mymodel", "1", run_id="run-1"),
            _FakeModelVersion("mymodel", "2", run_id="run-2"),
        ])

        service = RegistryService()
        versions = service.list_versions("mymodel")

        assert len(versions) == 2
        assert versions[0].version == "1"
        assert versions[1].version == "2"

    def test_get_version(self, monkeypatch):
        _patch_registry(monkeypatch, versions=[
            _FakeModelVersion("mymodel", "1", run_id="run-1", source="s3://bucket/model"),
        ])

        service = RegistryService()
        version = service.get_version("mymodel", "1")

        assert version.model_name == "mymodel"
        assert version.version == "1"
        assert version.run_id == "run-1"


class TestRegistryServiceRegister:
    def test_register_creates_model_and_version(self, monkeypatch):
        log = _patch_registry(monkeypatch)

        service = RegistryService()
        version = service.register_model(
            "new-model",
            source="file:///models/new-model/artifacts",
            run_id="run-42",
            description="Test model",
        )

        assert version.model_name == "new-model"
        assert version.version == "1"
        assert len(log.created_models) == 1
        assert len(log.created_versions) == 1

    def test_register_existing_model_creates_only_version(self, monkeypatch):
        log = _patch_registry(monkeypatch, models=[
            _FakeRegisteredModel("existing-model"),
        ])

        service = RegistryService()
        version = service.register_model(
            "existing-model",
            source="file:///models/existing-model/v2",
        )

        assert version.model_name == "existing-model"
        assert len(log.created_models) == 0
        assert len(log.created_versions) == 1


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


class TestPromotion:
    def test_promote_champion(self, monkeypatch):
        log = _patch_registry(monkeypatch, models=[
            _FakeRegisteredModel("mymodel"),
        ], versions=[
            _FakeModelVersion("mymodel", "1"),
        ])

        service = RegistryService()
        result = service.promote(PromotionRequest(
            model_name="mymodel",
            version="1",
            action=PromotionAction.CHAMPION,
        ))

        assert result.success is True
        assert any("champion" in c for c in result.alias_changes)
        assert log.alias_log == [("mymodel", "champion", "1")]
        assert ("mymodel", "1", "app.status", "champion") in log.version_tag_log

    def test_promote_candidate(self, monkeypatch):
        log = _patch_registry(monkeypatch, models=[
            _FakeRegisteredModel("mymodel"),
        ], versions=[
            _FakeModelVersion("mymodel", "2"),
        ])

        service = RegistryService()
        result = service.promote(PromotionRequest(
            model_name="mymodel",
            version="2",
            action=PromotionAction.CANDIDATE,
        ))

        assert result.success is True
        assert log.alias_log == [("mymodel", "candidate", "2")]
        assert ("mymodel", "2", "app.status", "candidate") in log.version_tag_log

    def test_promote_champion_clears_previous_status_tag(self, monkeypatch):
        log = _patch_registry(monkeypatch, models=[
            _FakeRegisteredModel("mymodel", aliases={"champion": "1"}),
        ], versions=[
            _FakeModelVersion("mymodel", "1", tags={"app.status": "champion"}),
            _FakeModelVersion("mymodel", "2"),
        ])

        service = RegistryService()
        result = service.promote(PromotionRequest(
            model_name="mymodel",
            version="2",
            action=PromotionAction.CHAMPION,
        ))

        assert result.success is True
        assert ("mymodel", "champion", "2") in log.alias_log
        assert ("mymodel", "1", "app.status") in log.deleted_version_tags
        assert ("mymodel", "2", "app.status", "champion") in log.version_tag_log

    def test_promote_archived(self, monkeypatch):
        log = _patch_registry(monkeypatch, models=[
            _FakeRegisteredModel("mymodel", aliases={"champion": "1"}),
        ], versions=[
            _FakeModelVersion("mymodel", "1"),
        ])

        service = RegistryService()
        result = service.promote(PromotionRequest(
            model_name="mymodel",
            version="1",
            action=PromotionAction.ARCHIVED,
        ))

        assert result.success is True
        assert ("mymodel", "1", "app.status", "archived") in log.version_tag_log
        assert ("mymodel", "champion") in log.deleted_aliases

    def test_promote_nonexistent_version_fails(self, monkeypatch):
        _patch_registry(monkeypatch, models=[
            _FakeRegisteredModel("mymodel"),
        ])

        service = RegistryService()
        with pytest.raises(PromotionError, match="not found"):
            service.promote(PromotionRequest(
                model_name="mymodel",
                version="999",
                action=PromotionAction.CHAMPION,
            ))


# ---------------------------------------------------------------------------
# Schema defaults
# ---------------------------------------------------------------------------


class TestRegistrySchemas:
    def test_promotion_request_model(self):
        request = PromotionRequest(
            model_name="test",
            version="1",
            action=PromotionAction.CHAMPION,
        )
        assert request.model_name == "test"
        assert request.action == PromotionAction.CHAMPION

    def test_promotion_result_defaults(self):
        result = PromotionResult(
            model_name="test",
            version="1",
            action=PromotionAction.CANDIDATE,
        )
        assert result.success is True
        assert result.alias_changes == []
        assert result.tag_changes == []
        assert result.warnings == []

    def test_registry_model_summary_defaults(self):
        summary = RegistryModelSummary(name="test")
        assert summary.version_count == 0
        assert summary.aliases == {}

    def test_registry_version_summary_defaults(self):
        summary = RegistryVersionSummary(model_name="test", version="1")
        assert summary.status == "UNKNOWN"
        assert summary.aliases == []
        assert summary.app_status is None


# ---------------------------------------------------------------------------
# CLI boundary
# ---------------------------------------------------------------------------


class TestRegistryCLI:
    def test_registry_list_cli(self, monkeypatch, capsys):
        from app import cli as cli_module

        monkeypatch.setattr(
            cli_module,
            "load_settings",
            lambda: type(
                "Settings",
                (),
                {"tracking": TrackingSettings()},
            )(),
        )
        monkeypatch.setattr(
            "app.tracking.mlflow_query.is_mlflow_available",
            lambda: True,
        )

        _patch_registry(monkeypatch, models=[
            _FakeRegisteredModel("model-x"),
        ])

        args = type("Args", (), {})()
        cli_module.cmd_registry_list(args)
        output = capsys.readouterr().out

        assert "model-x" in output

    def test_registry_promote_cli(self, monkeypatch, capsys):
        from app import cli as cli_module

        monkeypatch.setattr(
            cli_module,
            "load_settings",
            lambda: type(
                "Settings",
                (),
                {"tracking": TrackingSettings()},
            )(),
        )
        monkeypatch.setattr(
            "app.tracking.mlflow_query.is_mlflow_available",
            lambda: True,
        )

        _patch_registry(monkeypatch, models=[
            _FakeRegisteredModel("model-x"),
        ], versions=[
            _FakeModelVersion("model-x", "1"),
        ])

        args = type("Args", (), {
            "model_name": "model-x",
            "version": "1",
            "action": "champion",
        })()

        cli_module.cmd_registry_promote(args)
        output = capsys.readouterr().out

        assert "champion" in output
        assert "model-x" in output
