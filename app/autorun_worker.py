"""Subprocess entry point for persistent Auto Run jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

from app.autorun import AutoRunConfig, run_auto_run
from app.config.settings import load_settings
from app.storage import AppJobStatus, AppJobType, JobRecord, build_metadata_store


def main(request_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    job_id = str(request["job_id"])
    directory = request_path.parent
    settings = load_settings()
    store = build_metadata_store(settings)
    if store is None:
        raise RuntimeError("Metadata storage is unavailable.")
    record = store.get_job(job_id) or JobRecord(
        job_id=job_id, job_type=AppJobType.FLAML, status=AppJobStatus.RUNNING
    )
    try:
        _update(store, record, 10, "loading data")
        dataframe = pd.read_csv(request["dataset_path"])
        if (directory / "cancel.requested").exists():
            raise KeyboardInterrupt
        _update(store, record, 20, "training")
        result = run_auto_run(
            dataframe,
            AutoRunConfig.model_validate(request["config"]),
            artifacts_dir=settings.artifacts.root_dir,
            models_dir=settings.artifacts.models_dir,
            job_id=job_id,
        )
        record.status = AppJobStatus.SUCCESS
        record.primary_artifact_path = result.model_path
        record.summary_path = result.artifact_paths["evaluation"]
        record.metadata.update(
            {
                "progress": 100,
                "stage": "complete",
                "model_path": str(result.model_path),
                "metadata_path": str(result.metadata_path),
                "artifacts": {key: str(value) for key, value in result.artifact_paths.items()},
            }
        )
    except KeyboardInterrupt:
        record.status = AppJobStatus.CANCELLED
        record.metadata.update({"stage": "cancelled"})
    except Exception as exc:
        record.status = AppJobStatus.FAILED
        record.metadata.update({"stage": "failed", "error": str(exc)[:500]})
    record.updated_at = datetime.now(timezone.utc)
    store.record_job(record)
    return 0 if record.status == AppJobStatus.SUCCESS else 1


def _update(store, record: JobRecord, progress: int, stage: str) -> None:  # noqa: ANN001
    record.status = AppJobStatus.RUNNING
    record.metadata.update({"progress": progress, "stage": stage})
    record.updated_at = datetime.now(timezone.utc)
    store.record_job(record)


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
