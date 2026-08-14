"""Reproducibility metadata for trained models and exported bundles."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess  # nosec B404
from typing import Any

from pydantic import BaseModel, Field


class ProvenanceManifest(BaseModel):
    """Sanitized, portable description of how a model was produced."""

    schema_version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    app_version: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    job_id: str | None = None
    engine: str
    task_type: str
    target_column: str
    dataset_fingerprint: str
    dataset_schema_hash: str | None = None
    row_count: int
    column_count: int
    random_seed: int
    configuration: dict[str, Any] = Field(default_factory=dict)
    package_versions: dict[str, str] = Field(default_factory=dict)
    model_sha256: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_provenance(
    *,
    engine: str,
    task_type: str,
    target_column: str,
    dataset_fingerprint: str,
    row_count: int,
    column_count: int,
    random_seed: int,
    configuration: dict[str, Any] | None = None,
    dataset_schema_hash: str | None = None,
    job_id: str | None = None,
    model_path: Path | None = None,
    repo_root: Path | None = None,
) -> ProvenanceManifest:
    """Build a manifest without source locators, credentials, or raw rows."""

    return ProvenanceManifest(
        app_version=_package_version("autotabml-studio"),
        git_commit=_git_value(repo_root, ["rev-parse", "HEAD"]),
        git_dirty=_git_dirty(repo_root),
        job_id=job_id,
        engine=engine,
        task_type=task_type,
        target_column=target_column,
        dataset_fingerprint=dataset_fingerprint,
        dataset_schema_hash=dataset_schema_hash,
        row_count=row_count,
        column_count=column_count,
        random_seed=random_seed,
        configuration=_json_safe(configuration or {}),
        package_versions={name: _package_version(name) for name in _engine_packages(engine)},
        model_sha256=sha256_file(model_path) if model_path and model_path.is_file() else None,
    )


def write_provenance(manifest: ProvenanceManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _engine_packages(engine: str) -> list[str]:
    packages = ["autotabml-studio", "pandas", "scikit-learn"]
    mapping = {"pycaret": "pycaret", "flaml": "flaml", "lazypredict": "lazypredict"}
    if engine.lower() in mapping:
        packages.append(mapping[engine.lower()])
    return packages


def _git_value(repo_root: Path | None, args: list[str]) -> str | None:
    try:
        result = subprocess.run(  # nosec B603 B607
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True, timeout=2
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty(repo_root: Path | None) -> bool | None:
    value = _git_value(repo_root, ["status", "--porcelain", "--untracked-files=no"])
    return None if value is None else bool(value)


def _json_safe(value: Any) -> Any:
    return _sanitize(json.loads(json.dumps(value, default=str)))


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if any(marker in str(key).lower() for marker in ("secret", "token", "password", "api_key"))
                else _sanitize(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


__all__ = ["ProvenanceManifest", "build_provenance", "sha256_file", "write_provenance"]
