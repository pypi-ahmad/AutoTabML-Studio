"""Tests for CUDA / GPU detection utilities."""

from __future__ import annotations

from app.gpu import (
    _driver_probe,
    _torch_cuda_available,
    cuda_summary,
    is_cuda_available,
    resolve_use_gpu,
)


class TestIsCudaAvailable:
    def test_returns_bool(self):
        # Cannot assert True/False in a CI environment without a GPU.
        assert isinstance(is_cuda_available(), bool)


class TestCudaSummary:
    def test_returns_expected_keys(self):
        summary = cuda_summary()

        assert "cuda_available" in summary
        assert "device_name" in summary
        assert "device_count" in summary
        assert isinstance(summary["cuda_available"], bool)


class TestResolveUseGpu:
    def test_false_always_returns_false(self):
        assert resolve_use_gpu(False) is False

    def test_true_returns_true_when_cuda_available(self, monkeypatch):
        monkeypatch.setattr("app.gpu.is_cuda_available", lambda: True)

        assert resolve_use_gpu(True) is True

    def test_true_returns_false_when_cuda_unavailable(self, monkeypatch):
        monkeypatch.setattr("app.gpu.is_cuda_available", lambda: False)

        assert resolve_use_gpu(True) is False

    def test_force_returns_force_when_cuda_available(self, monkeypatch):
        monkeypatch.setattr("app.gpu.is_cuda_available", lambda: True)

        assert resolve_use_gpu("force") == "force"

    def test_force_preserves_force_when_cuda_unavailable(self, monkeypatch):
        monkeypatch.setattr("app.gpu.is_cuda_available", lambda: False)

        assert resolve_use_gpu("force") == "force"


class TestTorchCudaAvailable:
    def test_returns_none_when_torch_missing(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        assert _torch_cuda_available() is None


class TestDriverProbe:
    def test_returns_bool(self):
        assert isinstance(_driver_probe(), bool)
