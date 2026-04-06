from __future__ import annotations

import builtins

import pytest

from app.modeling.pycaret import setup_runner
from app.modeling.pycaret.errors import PyCaretDependencyError


def test_probe_pycaret_import_error_captures_runtimeerror(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"pycaret.classification", "pycaret.regression"}:
            raise RuntimeError("PyCaret only supports python 3.9, 3.10, 3.11.")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    error = setup_runner._probe_pycaret_import_error()

    assert isinstance(error, RuntimeError)
    assert "PyCaret only supports python 3.9, 3.10, 3.11." in str(error)


def test_is_pycaret_available_returns_false_for_runtime_import_failures(monkeypatch):
    monkeypatch.setattr(
        setup_runner,
        "_probe_pycaret_import_error",
        lambda: RuntimeError("unsupported interpreter"),
    )

    assert setup_runner.is_pycaret_available() is False


def test_require_pycaret_wraps_runtime_import_failures_with_guidance(monkeypatch):
    monkeypatch.setattr(
        setup_runner,
        "_probe_pycaret_import_error",
        lambda: RuntimeError(
            "PyCaret only supports python 3.9, 3.10, 3.11.",
            "Please DOWNGRADE your Python version.",
        ),
    )

    with pytest.raises(PyCaretDependencyError) as exc_info:
        setup_runner.require_pycaret()

    message = str(exc_info.value)
    assert "pycaret is not available in the current environment" in message
    assert "PyCaret only supports python 3.9, 3.10, 3.11." in message
    assert "Please DOWNGRADE your Python version." in message