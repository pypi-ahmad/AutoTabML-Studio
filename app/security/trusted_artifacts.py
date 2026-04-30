"""Trusted local model artifact helpers.

This module centralizes three controls for any local model artifact load:

1. Canonicalize every path before use.
2. Require the path to stay within an approved trust root.
3. Verify SHA256 checksums before deserializing any local artifact.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.security.errors import TrustedArtifactError

TRUSTED_MODEL_SOURCE = "autotabml_trusted_local_model_v1"
CHECKSUM_FILE_SUFFIX = ".sha256"


@dataclass(frozen=True)
class VerifiedArtifact:
    """Canonical artifact path plus the verified checksum metadata."""

    path: Path
    checksum: str
    checksum_path: Path


def compute_sha256(path: Path) -> str:
    """Return the SHA256 digest for a file on disk."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_file_path(path: Path) -> Path:
    """Return the checksum sidecar path for an artifact."""

    return path.with_name(f"{path.name}{CHECKSUM_FILE_SUFFIX}")


def write_checksum_file(path: Path, *, checksum: str | None = None) -> Path:
    """Write a SHA256 checksum sidecar for an artifact and return the sidecar path."""

    canonical_path = path.resolve(strict=True)
    checksum_path = checksum_file_path(canonical_path)
    checksum_path.write_text(f"{checksum or compute_sha256(canonical_path)}\n", encoding="utf-8")
    return checksum_path


def read_checksum_file(path: Path) -> str:
    """Read a checksum sidecar and return the normalized digest string."""

    canonical_path = path.resolve(strict=True)
    if not canonical_path.exists():
        raise TrustedArtifactError(f"Checksum file is missing: {canonical_path}")
    checksum = canonical_path.read_text(encoding="utf-8").strip().lower()
    if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise TrustedArtifactError(f"Checksum file is invalid: {canonical_path}")
    return checksum


def canonicalize_trusted_path(
    path: str | Path,
    *,
    trusted_roots: Iterable[Path],
    label: str,
) -> Path:
    """Resolve a path and ensure it remains under one of the approved roots."""

    raw_path = Path(path).expanduser()
    try:
        canonical_path = raw_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TrustedArtifactError(f"{label.title()} does not exist: {raw_path}") from exc

    normalized_roots = []
    for root in trusted_roots:
        resolved_root = Path(root).expanduser().resolve(strict=False)
        if resolved_root not in normalized_roots:
            normalized_roots.append(resolved_root)

    if not normalized_roots:
        raise TrustedArtifactError(f"No trusted roots configured for {label} validation.")

    for root in normalized_roots:
        if _is_relative_to(canonical_path, root):
            return canonical_path

    trusted_display = ", ".join(str(root) for root in normalized_roots)
    raise TrustedArtifactError(
        f"{label.title()} must remain inside a trusted directory. "
        f"Resolved path '{canonical_path}' is outside: {trusted_display}"
    )


def verify_local_artifact(
    path: str | Path,
    *,
    trusted_roots: Iterable[Path],
    expected_sha256: str | None = None,
    label: str = "artifact",
) -> VerifiedArtifact:
    """Validate trust root membership and SHA256 integrity for a local artifact."""

    canonical_path = canonicalize_trusted_path(path, trusted_roots=trusted_roots, label=label)
    checksum_path = canonicalize_trusted_path(
        checksum_file_path(canonical_path),
        trusted_roots=trusted_roots,
        label=f"{label} checksum",
    )
    sidecar_checksum = read_checksum_file(checksum_path)
    actual_checksum = compute_sha256(canonical_path)
    if actual_checksum != sidecar_checksum:
        raise TrustedArtifactError(
            f"{label.title()} checksum mismatch for '{canonical_path}'. "
            f"Expected {sidecar_checksum}, found {actual_checksum}."
        )

    if expected_sha256 is not None:
        normalized_expected = expected_sha256.strip().lower()
        if not normalized_expected:
            raise TrustedArtifactError(f"{label.title()} checksum metadata is blank.")
        if normalized_expected != sidecar_checksum:
            raise TrustedArtifactError(
                f"{label.title()} checksum does not match trusted metadata for '{canonical_path}'."
            )

    return VerifiedArtifact(path=canonical_path, checksum=sidecar_checksum, checksum_path=checksum_path)


def require_trusted_source(metadata: dict, *, artifact_label: str = "model") -> None:
    """Require the standardized trusted-source marker in persisted metadata."""

    trusted_source = str(metadata.get("trusted_source") or "").strip()
    if trusted_source != TRUSTED_MODEL_SOURCE:
        raise TrustedArtifactError(
            f"{artifact_label.title()} metadata is missing the trusted source marker. "
            "Re-save the model from within AutoTabML Studio to regenerate trusted metadata."
        )


def require_metadata_checksum(metadata: dict, *, field_name: str = "model_sha256") -> str:
    """Return a required checksum field from metadata, raising when absent."""

    checksum = str(metadata.get(field_name) or "").strip().lower()
    if not checksum:
        raise TrustedArtifactError(
            f"Model metadata is missing required checksum field '{field_name}'. "
            "Re-save the model from within AutoTabML Studio to regenerate trusted metadata."
        )
    return checksum


def load_verified_pickle_artifact(
    path: str | Path,
    *,
    trusted_roots: Iterable[Path],
    expected_sha256: str,
):
    """Load a pickle artifact only after trust-root and checksum validation."""

    verified = verify_local_artifact(
        path,
        trusted_roots=trusted_roots,
        expected_sha256=expected_sha256,
        label="model artifact",
    )
    with verified.path.open("rb") as handle:
        return pickle.load(handle)


def load_verified_skops_artifact(
    path: str | Path,
    *,
    trusted_roots: Iterable[Path],
    expected_sha256: str,
    trusted_types: list[str] | None = None,
):
    """Load a skops artifact only after trust-root and checksum validation."""

    verified = verify_local_artifact(
        path,
        trusted_roots=trusted_roots,
        expected_sha256=expected_sha256,
        label="model artifact",
    )
    try:
        from skops.io import load as skops_load
    except ImportError as exc:
        raise TrustedArtifactError(
            "Secure sklearn model loading requires the 'skops' package. Install the benchmark extras to load this model."
        ) from exc

    return skops_load(verified.path, trusted=trusted_types or [])


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False