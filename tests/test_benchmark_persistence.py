"""Focused tests for trusted benchmark model persistence helpers."""

from __future__ import annotations

from pathlib import Path

from app.modeling.benchmark.persistence import discover_saved_benchmark_models, load_saved_benchmark_model
from app.modeling.benchmark.schemas import BenchmarkSavedModelMetadata, BenchmarkTaskType
from app.security.trusted_artifacts import TRUSTED_MODEL_SOURCE, compute_sha256, write_checksum_file


class TestBenchmarkPersistence:
    def test_discover_saved_benchmark_models_returns_trusted_metadata(self, tmp_path: Path):
        model_path = tmp_path / "benchmark_model.skops"
        model_path.write_bytes(b"benchmark-model")
        model_sha256 = compute_sha256(model_path)
        write_checksum_file(model_path, checksum=model_sha256)

        metadata = BenchmarkSavedModelMetadata(
            model_name="benchmark_model",
            task_type=BenchmarkTaskType.CLASSIFICATION,
            target_column="target",
            model_path=model_path,
            artifact_format="skops",
            trusted_source=TRUSTED_MODEL_SOURCE,
            model_sha256=model_sha256,
            trusted_types=["sklearn.linear_model._logistic.LogisticRegression"],
        )
        metadata_path = tmp_path / "benchmark_model.json"
        metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        write_checksum_file(metadata_path)

        discovered = discover_saved_benchmark_models(tmp_path)

        assert len(discovered) == 1
        assert discovered[0]["model_name"] == "benchmark_model"
        assert discovered[0]["artifact_format"] == "skops"

    def test_discover_saved_benchmark_models_ignores_untrusted_metadata(self, tmp_path: Path):
        model_path = tmp_path / "benchmark_model.skops"
        model_path.write_bytes(b"benchmark-model")
        metadata = BenchmarkSavedModelMetadata(
            model_name="benchmark_model",
            task_type=BenchmarkTaskType.CLASSIFICATION,
            target_column="target",
            model_path=model_path,
            artifact_format="skops",
        )
        metadata_path = tmp_path / "benchmark_model.json"
        metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

        discovered = discover_saved_benchmark_models(tmp_path)

        assert discovered == []

    def test_load_saved_benchmark_model_passes_trusted_types(self, tmp_path: Path, monkeypatch):
        model_path = tmp_path / "benchmark_model.skops"
        model_path.write_bytes(b"benchmark-model")
        metadata = BenchmarkSavedModelMetadata(
            model_name="benchmark_model",
            task_type=BenchmarkTaskType.CLASSIFICATION,
            target_column="target",
            model_path=model_path,
            artifact_format="skops",
            trusted_source=TRUSTED_MODEL_SOURCE,
            model_sha256=compute_sha256(model_path),
            trusted_types=["trusted.Type"],
        )

        captured: dict = {}

        def _fake_load(path, *, trusted_roots, expected_sha256, trusted_types):
            captured.update(
                {
                    "path": path,
                    "trusted_roots": trusted_roots,
                    "expected_sha256": expected_sha256,
                    "trusted_types": trusted_types,
                }
            )
            return {"loaded": True}

        monkeypatch.setattr("app.modeling.benchmark.persistence.load_verified_skops_artifact", _fake_load)

        loaded = load_saved_benchmark_model(metadata, trusted_roots=[tmp_path])

        assert loaded == {"loaded": True}
        assert captured["path"] == model_path
        assert captured["trusted_roots"] == [tmp_path]
        assert captured["expected_sha256"] == metadata.model_sha256
        assert captured["trusted_types"] == ["trusted.Type"]