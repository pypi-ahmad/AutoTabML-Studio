"""Compatibility shim for trusted artifact helpers.

Import from app.security.trusted_artifacts in new code.
"""

from app.security.trusted_artifacts import (
    CHECKSUM_FILE_SUFFIX,
    TRUSTED_MODEL_SOURCE,
    VerifiedArtifact,
    canonicalize_trusted_path,
    checksum_file_path,
    compute_sha256,
    load_verified_pickle_artifact,
    load_verified_skops_artifact,
    read_checksum_file,
    require_metadata_checksum,
    require_trusted_source,
    verify_local_artifact,
    write_checksum_file,
)

__all__ = [
    "CHECKSUM_FILE_SUFFIX",
    "TRUSTED_MODEL_SOURCE",
    "VerifiedArtifact",
    "canonicalize_trusted_path",
    "checksum_file_path",
    "compute_sha256",
    "load_verified_pickle_artifact",
    "load_verified_skops_artifact",
    "read_checksum_file",
    "require_metadata_checksum",
    "require_trusted_source",
    "verify_local_artifact",
    "write_checksum_file",
]