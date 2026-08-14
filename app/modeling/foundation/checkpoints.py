"""Pinned Hugging Face checkpoint resolution with explicit download consent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointSpec:
    """Immutable model repository identity."""

    repo_id: str
    revision: str


TABFM_CHECKPOINT = CheckpointSpec(
    repo_id="google/tabfm-1.0.0-pytorch",
    revision="77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83",
)
TIMESFM_CHECKPOINT = CheckpointSpec(
    repo_id="google/timesfm-2.5-200m-pytorch",
    revision="1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
)


class ModelDownloadRequiredError(RuntimeError):
    """Raised when a pinned checkpoint is not cached and download was not approved."""


class CheckpointResolver:
    """Resolve a pinned snapshot from cache before considering network access."""

    def __init__(self, snapshot_download: Callable[..., str] | None = None) -> None:
        self._snapshot_download = snapshot_download

    def resolve(
        self,
        spec: CheckpointSpec,
        *,
        allow_download: bool,
        cache_dir: Path | None = None,
    ) -> Path:
        download = self._snapshot_download or _snapshot_download
        common: dict[str, Any] = {
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "cache_dir": str(cache_dir) if cache_dir else None,
        }
        try:
            return Path(download(**common, local_files_only=True))
        except Exception as exc:
            if not allow_download:
                raise ModelDownloadRequiredError(
                    f"Checkpoint {spec.repo_id}@{spec.revision} is not cached. Confirm the first download to continue."
                ) from exc
        return Path(download(**common, local_files_only=False))


def _snapshot_download(**kwargs: Any) -> str:
    try:
        snapshot_download = import_module("huggingface_hub").snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Install the foundation-model extra to use Google foundation models.") from exc
    return snapshot_download(**kwargs)
