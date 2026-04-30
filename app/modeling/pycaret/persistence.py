"""Finalize, save, and load helpers for experiment artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.artifacts import ArtifactKind, LocalArtifactManager
from app.modeling.pycaret.schemas import ExperimentTaskType, SavedModelMetadata
from app.modeling.pycaret.setup_runner import build_pycaret_experiment
from app.path_utils import safe_artifact_stem
from app.security.trusted_artifacts import TRUSTED_MODEL_SOURCE, compute_sha256, write_checksum_file


def build_saved_model_metadata(
    *,
    task_type: ExperimentTaskType,
    target_column: str,
    model_id: str | None,
    model_name: str,
    model_path: Path,
    dataset_name: str | None = None,
    dataset_fingerprint: str | None,
    feature_columns: list[str],
    feature_dtypes: dict[str, str],
    target_dtype: str | None,
    experiment_snapshot_path: Path | None,
    model_only: bool = False,
    artifact_format: str | None = None,
    trusted_source: str | None = None,
    model_sha256: str | None = None,
    experiment_snapshot_sha256: str | None = None,
) -> SavedModelMetadata:
    """Create stable metadata for a saved PyCaret model artifact."""

    return SavedModelMetadata(
        task_type=task_type,
        target_column=target_column,
        model_id=model_id,
        model_name=model_name,
        model_path=model_path,
        dataset_name=dataset_name,
        dataset_fingerprint=dataset_fingerprint,
        trained_at=datetime.now(timezone.utc).isoformat(),
        feature_columns=feature_columns,
        feature_dtypes=feature_dtypes,
        target_dtype=target_dtype,
        experiment_snapshot_path=experiment_snapshot_path,
        experiment_snapshot_includes_data=False,
        model_only=model_only,
        artifact_format=artifact_format,
        trusted_source=trusted_source,
        model_sha256=model_sha256,
        experiment_snapshot_sha256=experiment_snapshot_sha256,
    )


def save_finalized_model(
    experiment_handle,
    finalized_model,
    *,
    task_type: ExperimentTaskType,
    target_column: str,
    model_id: str | None,
    model_name: str,
    save_name: str,
    models_dir: Path,
    snapshots_dir: Path,
    dataset_name: str | None = None,
    dataset_fingerprint: str | None,
    feature_columns: list[str],
    feature_dtypes: dict[str, str],
    target_dtype: str | None,
    save_experiment_snapshot: bool,
    model_only: bool,
) -> SavedModelMetadata:  # noqa: ANN001
    """Persist a finalized model and optional experiment snapshot."""

    models_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    base_name = safe_artifact_stem(save_name or model_name)
    model_base_path = models_dir / base_name
    _, saved_path = experiment_handle.save_model(
        finalized_model,
        str(model_base_path),
        model_only=model_only,
        verbose=False,
    )
    saved_model_path = Path(saved_path)
    model_sha256 = compute_sha256(saved_model_path)
    write_checksum_file(saved_model_path, checksum=model_sha256)

    snapshot_path: Path | None = None
    snapshot_sha256: str | None = None
    if save_experiment_snapshot:
        snapshot_path = snapshots_dir / f"{base_name}_experiment.pkl"
        experiment_handle.save_experiment(snapshot_path)
        snapshot_sha256 = compute_sha256(snapshot_path)
        write_checksum_file(snapshot_path, checksum=snapshot_sha256)

    return build_saved_model_metadata(
        task_type=task_type,
        target_column=target_column,
        model_id=model_id,
        model_name=model_name,
        model_path=saved_model_path,
        dataset_name=dataset_name,
        dataset_fingerprint=dataset_fingerprint,
        feature_columns=feature_columns,
        feature_dtypes=feature_dtypes,
        target_dtype=target_dtype,
        experiment_snapshot_path=snapshot_path,
        model_only=model_only,
        artifact_format="pycaret_pickle",
        trusted_source=TRUSTED_MODEL_SOURCE,
        model_sha256=model_sha256,
        experiment_snapshot_sha256=snapshot_sha256,
    )


def write_saved_model_metadata_sidecar(
    metadata: SavedModelMetadata,
    *,
    output_dir: Path,
    stem: str | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Persist one saved-model metadata sidecar for later prediction discovery."""

    manager = LocalArtifactManager()
    metadata_path = manager.build_artifact_path(
        kind=ArtifactKind.MODEL,
        stem=stem or metadata.model_name,
        label="saved_model_metadata",
        suffix=".json",
        timestamp=timestamp,
        output_dir=output_dir,
        ensure_unique=True,
    )
    manager.write_text(metadata_path, metadata.model_dump_json(indent=2))
    write_checksum_file(metadata_path)
    return metadata_path


def load_model_artifact(task_type: ExperimentTaskType, model_name_or_path: str | Path):
    """Load a saved model artifact without rebuilding a full setup run."""

    experiment_handle = build_pycaret_experiment(task_type)
    path = Path(model_name_or_path)
    if path.suffix == ".pkl":
        path = path.with_suffix("")
    return experiment_handle.load_model(str(path), verbose=False)


def load_experiment_snapshot(
    task_type: ExperimentTaskType,
    snapshot_path: str | Path,
    *,
    data,
    test_data=None,
):
    """Load a previously saved experiment snapshot with explicit data."""

    experiment_handle = build_pycaret_experiment(task_type)
    return experiment_handle.load_experiment(
        snapshot_path,
        data=data,
        test_data=test_data,
    )