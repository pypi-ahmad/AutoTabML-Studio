"""Tests for centralized page registration and dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config.enums import WorkspaceMode
from app.pages.registry import default_page_label, get_page_by_label, get_page_registry, render_registered_page


class TestPageRegistry:
    def test_default_page_label_matches_workspace_mode(self):
        assert default_page_label(WorkspaceMode.DASHBOARD) == "Home"
        assert default_page_label(WorkspaceMode.NOTEBOOK) == "Notebook"

    def test_get_page_by_label_raises_for_unknown_page(self):
        with pytest.raises(KeyError, match="Unknown page label"):
            get_page_by_label("Missing")

    def test_render_registered_page_imports_and_calls_render_function(self, monkeypatch):
        called = {"rendered": False}

        def _render_dashboard_page():
            called["rendered"] = True

        monkeypatch.setattr(
            "app.pages.registry.importlib.import_module",
            lambda module_path: SimpleNamespace(render_dashboard_page=_render_dashboard_page),
        )

        render_registered_page("Home")

        assert called["rendered"] is True

    def test_page_registry_labels_are_unique(self):
        labels = [page.label for page in get_page_registry()]

        assert len(labels) == len(set(labels))

    def test_dataset_intake_page_is_registered(self):
        labels = [page.label for page in get_page_registry()]

        assert "Load Data" in labels