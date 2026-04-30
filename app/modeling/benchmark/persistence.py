"""Trusted benchmark model discovery and loading helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from app.errors import log_exception
from app.modeling.benchmark.schemas import BenchmarkSavedModelMetadata
from app.security.errors import TrustedArtifactError
from app.security.trusted_artifacts import (
    load_verified_skops_artifact,
    require_metadata_checksum,
    require_trusted_source,
    verify_local_artifact,
)

logger = logging.getLogger(__name__)


def load_saved_benchmark_model_metadata_file(
    path: Path,
    *,
    trusted_roots: list[Path],
    raise_on_error: bool = False,
) -> BenchmarkSavedModelMetadata | None:
    """Parse a benchmark saved-model metadata file, returning None when invalid."""

    try:
        verified_metadata = verify_local_artifact(path, trusted_roots=trusted_roots, label="benchmark metadata")
        metadata = BenchmarkSavedModelMetadata.model_validate_json(verified_metadata.path.read_text(encoding="utf-8"))
        payload = metadata.model_dump(mode="json")
        if payload.get("source") != "benchmark":
            raise TrustedArtifactError("Metadata does not describe a benchmark-saved model.")
        require_trusted_source(payload, artifact_label="benchmark model")
        if metadata.artifact_format != "skops":
            raise TrustedArtifactError(
                "Benchmark models must use the trusted skops format. Re-save the model from Quick Benchmark."
            )
        model_sha256 = require_metadata_checksum(payload)
        verified_model = verify_local_artifact(
            metadata.model_path,
            trusted_roots=trusted_roots,
            expected_sha256=model_sha256,
            label="benchmark model artifact",
        )
        return metadata.model_copy(update={"model_path": verified_model.path})
    except (TrustedArtifactError, ValidationError, OSError, ValueError, json.JSONDecodeError) as exc:
        if raise_on_error:
            raise
        log_exception(
            logger,
            exc,
            operation="benchmark.load_saved_metadata",
            level=logging.DEBUG,
            context={"metadata_path": str(path)},
        )
        return None


def discover_saved_benchmark_models(models_dir: Path) -> list[dict]:
    """Discover trusted benchmark models from checksum-backed metadata sidecars."""

    results: list[dict] = []
    if not models_dir.exists():
        return results

    for metadata_path in models_dir.glob("*.json"):
        metadata = load_saved_benchmark_model_metadata_file(metadata_path, trusted_roots=[models_dir])
        if metadata is None:
            continue
        payload = metadata.model_dump(mode="json")
        payload["_model_path"] = str(metadata.model_path)
        payload["_metadata_path"] = str(metadata_path.resolve())
        results.append(payload)

    return sorted(results, key=lambda item: str(item.get("model_name", "")).lower())


def load_saved_benchmark_model(
    metadata: BenchmarkSavedModelMetadata,
    *,
    trusted_roots: list[Path],
):
    """Load a trusted benchmark model from a skops artifact."""

    return load_verified_skops_artifact(
        metadata.model_path,
        trusted_roots=trusted_roots,
        expected_sha256=require_metadata_checksum(metadata.model_dump(mode="json")),
        trusted_types=list(metadata.trusted_types),
    )