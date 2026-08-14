"""Persistent local subprocess jobs for long-running AutoML work."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from uuid import uuid4

import pandas as pd

from app.autorun import AutoRunConfig
from app.storage import AppJobStatus, AppJobType, AppMetadataStore, JobRecord


class BackgroundJobService:
    """Submit and control one local training process at a time."""

    def __init__(self, *, store: AppMetadataStore, jobs_dir: Path) -> None:
        self.store = store
        self.jobs_dir = jobs_dir

    def submit_auto_run(self, dataframe: pd.DataFrame, config: AutoRunConfig) -> JobRecord:
        active = [
            job
            for job in self.store.list_recent_jobs(limit=100)
            if job.status in {AppJobStatus.QUEUED, AppJobStatus.RUNNING, AppJobStatus.CANCEL_REQUESTED}
        ]
        if active:
            raise RuntimeError("Another training job is already active.")
        job_id = f"autorun-{uuid4().hex[:12]}"
        directory = self.jobs_dir / job_id
        directory.mkdir(parents=True, exist_ok=False)
        dataset_path = directory / "dataset.csv"
        dataframe.to_csv(dataset_path, index=False)
        request_path = directory / "request.json"
        request_path.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "dataset_path": str(dataset_path.resolve()),
                    "config": config.model_dump(mode="json"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        record = JobRecord(
            job_id=job_id,
            job_type=AppJobType.FLAML,
            status=AppJobStatus.QUEUED,
            title=f"Auto Run · {config.model_name}",
            metadata={"progress": 0, "stage": "queued", "job_dir": str(directory.resolve())},
        )
        self.store.record_job(record)
        process = subprocess.Popen(
            [sys.executable, "-m", "app.autorun_worker", str(request_path.resolve())],
            cwd=Path.cwd(),
            stdout=(directory / "worker.log").open("a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        record.status = AppJobStatus.RUNNING
        record.metadata.update({"pid": process.pid, "progress": 5, "stage": "starting"})
        record.updated_at = datetime.now(timezone.utc)
        self.store.record_job(record)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        return self.store.get_job(job_id)

    def cancel(self, job_id: str) -> JobRecord:
        record = self.store.get_job(job_id)
        if record is None:
            raise KeyError(job_id)
        if record.status not in {AppJobStatus.QUEUED, AppJobStatus.RUNNING}:
            return record
        record.status = AppJobStatus.CANCEL_REQUESTED
        directory = Path(str(record.metadata["job_dir"]))
        (directory / "cancel.requested").touch()
        pid = int(record.metadata.get("pid", 0))
        if pid > 0:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        record.status = AppJobStatus.CANCELLED
        record.metadata.update({"stage": "cancelled"})
        record.updated_at = datetime.now(timezone.utc)
        self.store.record_job(record)
        return record


__all__ = ["BackgroundJobService"]
