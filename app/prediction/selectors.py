"""Discovery and source-selection helpers for prediction flows."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from app.errors import log_exception
from app.modeling.pycaret.schemas import ExperimentTaskType, SavedModelMetadata
from app.prediction.errors import ModelDiscoveryError
from app.prediction.schemas import AvailableModelReference, ModelSourceType, PredictionTaskType
from app.security.errors import TrustedArtifactError
from app.security.trusted_artifacts import require_metadata_checksum, require_trusted_source, verify_local_artifact

logger = logging.getLogger(__name__)

_SAVED_MODEL_METADATA_GLOB = "*_saved_model_metadata_*.json"
_FLAML_METADATA_GLOB = "*_flaml_saved_model_metadata_*.json"


def discover_local_saved_models(
    model_dirs: list[Path],
    metadata_dirs: list[Path],
) -> list[AvailableModelReference]:
    """Discover local saved models and optional metadata sidecars."""

    references_by_path: dict[str, tuple[SavedModelMetadata, Path]] = {}
    trusted_metadata_dirs = _dedupe_paths(metadata_dirs + model_dirs)

    for metadata_dir in trusted_metadata_dirs:
        if not metadata_dir.exists():
            continue
        for metadata_path in metadata_dir.rglob(_SAVED_MODEL_METADATA_GLOB):
            metadata = load_saved_model_metadata_file(
                metadata_path,
                metadata_roots=trusted_metadata_dirs,
                model_roots=model_dirs,
            )
            if metadata is None:
                continue
            model_path = Path(metadata.model_path)
            key = _path_key(model_path)
            current = references_by_path.get(key)
            if current is None or metadata_path.stat().st_mtime > current[1].stat().st_mtime:
                references_by_path[key] = (metadata, metadata_path)

    references: list[AvailableModelReference] = []
    known_paths = set()

    for metadata, metadata_path in references_by_path.values():
        model_path = Path(metadata.model_path)
        known_paths.add(_path_key(model_path))
        references.append(
            AvailableModelReference(
                source_type=ModelSourceType.LOCAL_SAVED_MODEL,
                display_name=metadata.model_name,
                model_identifier=metadata.model_name,
                load_reference=str(model_path),
                task_type=coerce_prediction_task_type(metadata.task_type),
                description="Local saved PyCaret model with saved metadata.",
                model_path=model_path,
                metadata_path=metadata_path,
                feature_columns=list(metadata.feature_columns),
                metadata={
                    "target_column": metadata.target_column,
                    "dataset_name": metadata.dataset_name,
                    "dataset_fingerprint": metadata.dataset_fingerprint,
                    "feature_dtypes": dict(metadata.feature_dtypes),
                    "target_dtype": metadata.target_dtype,
                    "model_only": metadata.model_only,
                    "trained_at": metadata.trained_at,
                    "artifact_format": metadata.artifact_format,
                    "trusted_source": metadata.trusted_source,
                },
            )
        )

    return sorted(references, key=lambda item: (item.display_name.lower(), item.load_reference.lower()))


def resolve_local_model_reference(
    identifier: str | Path,
    references: list[AvailableModelReference],
) -> AvailableModelReference:
    """Resolve one local model reference from a user-supplied identifier."""

    candidate = str(identifier).strip()
    if not candidate:
        raise ModelDiscoveryError("Model identifier must not be blank.")

    path = Path(candidate)
    normalized_path = _path_key(path)
    normalized_without_suffix = _path_key(path.with_suffix(""))

    matches = []
    for reference in references:
        options = {
            reference.model_identifier.lower(),
            reference.display_name.lower(),
            reference.load_reference.lower(),
            Path(reference.load_reference).stem.lower(),
            _path_key(Path(reference.load_reference)).lower(),
            _path_key(Path(reference.load_reference).with_suffix("")).lower(),
        }
        if candidate.lower() in options or normalized_path.lower() in options or normalized_without_suffix.lower() in options:
            matches.append(reference)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ModelDiscoveryError(
            f"Model identifier '{candidate}' matched multiple local saved models. Use an explicit path."
        )

    if path.exists() or path.with_suffix(".pkl").exists() or path.with_suffix(".skops").exists():
        raise ModelDiscoveryError(
            "Local model paths must resolve to a trusted saved model discovered from checksum-backed metadata. "
            "Re-save the model from within AutoTabML Studio before loading it for prediction."
        )

    raise ModelDiscoveryError(f"Could not resolve local saved model '{candidate}'.")


def load_saved_model_metadata_file(
    path: Path,
    *,
    metadata_roots: list[Path],
    model_roots: list[Path],
    raise_on_error: bool = False,
) -> SavedModelMetadata | None:
    """Parse a saved-model metadata file, returning None when invalid."""

    try:
        verified_metadata = verify_local_artifact(path, trusted_roots=metadata_roots, label="model metadata")
        metadata = SavedModelMetadata.model_validate_json(verified_metadata.path.read_text(encoding="utf-8"))
        payload = metadata.model_dump(mode="json")
        require_trusted_source(payload)
        model_sha256 = require_metadata_checksum(payload)
        verified_model = verify_local_artifact(
            metadata.model_path,
            trusted_roots=model_roots,
            expected_sha256=model_sha256,
            label="model artifact",
        )
        return metadata.model_copy(update={"model_path": verified_model.path})
    except (TrustedArtifactError, ValidationError, OSError, ValueError, json.JSONDecodeError) as exc:
        if raise_on_error:
            raise
        log_exception(
            logger,
            exc,
            operation="prediction.load_saved_model_metadata",
            level=logging.DEBUG,
            context={"metadata_path": str(path)},
        )
        return None


def build_mlflow_run_model_uri(run_id: str, artifact_path: str) -> str:
    """Build a runs:/ URI from an MLflow run id and artifact path."""

    cleaned_run_id = run_id.strip()
    cleaned_artifact_path = artifact_path.strip().lstrip("/")
    if not cleaned_run_id or not cleaned_artifact_path:
        raise ModelDiscoveryError("MLflow run model URIs require both run_id and artifact_path.")
    return f"runs:/{cleaned_run_id}/{cleaned_artifact_path}"


def build_mlflow_registered_model_uri(
    model_name: str,
    *,
    version: str | None = None,
    alias: str | None = None,
) -> str:
    """Build a models:/ URI for a registered model."""

    cleaned_name = model_name.strip()
    if not cleaned_name:
        raise ModelDiscoveryError("Registered-model URIs require a model name.")
    if version and alias:
        raise ModelDiscoveryError("Specify either version or alias for a registered model URI, not both.")
    if alias:
        return f"models:/{cleaned_name}@{alias.strip()}"
    if version:
        return f"models:/{cleaned_name}/{version.strip()}"
    raise ModelDiscoveryError("Registered-model URIs require a version or alias.")


def extract_run_id_from_model_uri(model_uri: str) -> str | None:
    """Return the MLflow run id from a runs:/ model URI when present."""

    if not model_uri.startswith("runs:/"):
        return None
    parts = model_uri.split("/")
    if len(parts) < 2:
        return None
    return parts[1] or None


def coerce_prediction_task_type(value) -> PredictionTaskType:  # noqa: ANN001
    """Convert supported task-type values into PredictionTaskType."""

    if isinstance(value, PredictionTaskType):
        return value
    if isinstance(value, ExperimentTaskType):
        if value == ExperimentTaskType.CLASSIFICATION:
            return PredictionTaskType.CLASSIFICATION
        if value == ExperimentTaskType.REGRESSION:
            return PredictionTaskType.REGRESSION
        return PredictionTaskType.UNKNOWN
    normalized = str(value or "").strip().lower()
    if normalized == PredictionTaskType.CLASSIFICATION.value:
        return PredictionTaskType.CLASSIFICATION
    if normalized == PredictionTaskType.REGRESSION.value:
        return PredictionTaskType.REGRESSION
    return PredictionTaskType.UNKNOWN


def to_experiment_task_type(task_type: PredictionTaskType) -> ExperimentTaskType:
    """Convert PredictionTaskType into the PyCaret experiment enum."""

    if task_type == PredictionTaskType.CLASSIFICATION:
        return ExperimentTaskType.CLASSIFICATION
    if task_type == PredictionTaskType.REGRESSION:
        return ExperimentTaskType.REGRESSION
    raise ModelDiscoveryError(
        "PyCaret model loading requires a known task type (classification or regression)."
    )


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except FileNotFoundError:
        return str(path)


# ── FLAML model discovery ────────────────────────────────────────────

def discover_flaml_saved_models(
    model_dirs: list[Path],
    metadata_dirs: list[Path],
) -> list[AvailableModelReference]:
    """Discover locally saved FLAML models using metadata sidecars."""

    from app.modeling.flaml.schemas import FlamlSavedModelMetadata

    references_by_path: dict[str, tuple[FlamlSavedModelMetadata, Path]] = {}

    search_dirs = _dedupe_paths(metadata_dirs + model_dirs)
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for metadata_path in search_dir.rglob(_FLAML_METADATA_GLOB):
            metadata = load_flaml_saved_model_metadata_file(
                metadata_path,
                metadata_roots=search_dirs,
                model_roots=model_dirs,
            )
            if metadata is None:
                continue
            model_path = Path(metadata.model_path)
            key = _path_key(model_path)
            current = references_by_path.get(key)
            if current is None or metadata_path.stat().st_mtime > current[1].stat().st_mtime:
                references_by_path[key] = (metadata, metadata_path)

    references: list[AvailableModelReference] = []
    for metadata, metadata_path in references_by_path.values():
        model_path = Path(metadata.model_path)
        task_type = _coerce_flaml_task_type(metadata.task_type)
        references.append(
            AvailableModelReference(
                source_type=ModelSourceType.LOCAL_SAVED_MODEL,
                display_name=metadata.model_name,
                model_identifier=metadata.model_name,
                load_reference=str(model_path),
                task_type=task_type,
                description="Local saved FLAML model with saved metadata.",
                model_path=model_path,
                metadata_path=metadata_path,
                feature_columns=list(metadata.feature_columns),
                metadata={
                    "framework": "flaml",
                    "target_column": metadata.target_column,
                    "dataset_name": metadata.dataset_name,
                    "dataset_fingerprint": metadata.dataset_fingerprint,
                    "feature_dtypes": dict(metadata.feature_dtypes),
                    "target_dtype": metadata.target_dtype,
                    "best_estimator": metadata.best_estimator,
                    "best_loss": metadata.best_loss,
                    "metric": metadata.metric,
                    "trained_at": metadata.trained_at,
                    "artifact_format": metadata.artifact_format,
                    "trusted_source": metadata.trusted_source,
                },
            )
        )

    return sorted(references, key=lambda r: (r.display_name.lower(), r.load_reference.lower()))


def load_flaml_saved_model_metadata_file(  # noqa: ANN201
    path: Path,
    *,
    metadata_roots: list[Path],
    model_roots: list[Path],
    raise_on_error: bool = False,
):
    """Parse a FLAML saved-model metadata file, returning None when invalid."""

    try:
        from app.modeling.flaml.schemas import FlamlSavedModelMetadata

        verified_metadata = verify_local_artifact(path, trusted_roots=metadata_roots, label="FLAML metadata")
        metadata = FlamlSavedModelMetadata.model_validate_json(verified_metadata.path.read_text(encoding="utf-8"))
        payload = metadata.model_dump(mode="json")
        require_trusted_source(payload, artifact_label="flaml model")
        model_sha256 = require_metadata_checksum(payload)
        verified_model = verify_local_artifact(
            metadata.model_path,
            trusted_roots=model_roots,
            expected_sha256=model_sha256,
            label="FLAML model artifact",
        )
        return metadata.model_copy(update={"model_path": verified_model.path})
    except (TrustedArtifactError, ValidationError, OSError, ValueError, json.JSONDecodeError) as exc:
        if raise_on_error:
            raise
        log_exception(
            logger,
            exc,
            operation="prediction.load_flaml_metadata",
            level=logging.DEBUG,
            context={"metadata_path": str(path)},
        )
        return None


def _coerce_flaml_task_type(value) -> PredictionTaskType:  # noqa: ANN001
    """Convert a FLAML task type to PredictionTaskType."""

    normalized = str(value.value if hasattr(value, "value") else value).strip().lower()
    if normalized == "classification":
        return PredictionTaskType.CLASSIFICATION
    if normalized == "regression":
        return PredictionTaskType.REGRESSION
    return PredictionTaskType.UNKNOWN


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique_paths: list[Path] = []
    for path in paths:
        if path not in unique_paths:
            unique_paths.append(path)
    return unique_paths
