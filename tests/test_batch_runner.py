from __future__ import annotations

import logging
from pathlib import Path

from app.storage.models import BatchItemStatus, BatchRunItemRecord


def test_declared_target_from_item_prefers_item_id_suffix_for_resume_keys():
    from scripts import batch_uci_runner as runner

    item = BatchRunItemRecord(
        item_id="batch-1::42::Type",
        batch_id="batch-1",
        uci_id=42,
        dataset_name="Glass Identification",
        target_column="Type_of_glass",
        status=BatchItemStatus.SUCCESS,
    )

    assert runner._declared_target_from_item(item) == "Type"


def test_should_skip_completed_dataset_supports_legacy_single_variant_batches():
    from scripts import batch_uci_runner as runner

    legacy_item = BatchRunItemRecord(
        item_id="batch-1::42",
        batch_id="batch-1",
        uci_id=42,
        dataset_name="Glass Identification",
        target_column="Type_of_glass",
        status=BatchItemStatus.SUCCESS,
    )

    should_skip = runner._should_skip_completed_dataset(
        42,
        "Type",
        items_by_uci={42: [legacy_item]},
        completed_keys=set(),
        variant_counts=runner.Counter({42: 1}),
    )

    assert should_skip is True


def test_setup_batch_logger_deduplicates_file_handlers(tmp_path: Path):
    from scripts import batch_uci_runner as runner

    log_path = tmp_path / "batch.log"
    batch_logger = logging.getLogger("batch_runner")
    app_logger = logging.getLogger("app")
    original_batch_handlers = list(batch_logger.handlers)
    original_app_handlers = list(app_logger.handlers)

    try:
        batch_logger.handlers = []
        app_logger.handlers = []

        runner._setup_batch_logger(log_path)
        runner._setup_batch_logger(log_path)

        batch_file_handlers = [
            handler
            for handler in batch_logger.handlers
            if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == log_path.resolve()
        ]
        app_file_handlers = [
            handler
            for handler in app_logger.handlers
            if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == log_path.resolve()
        ]

        assert len(batch_file_handlers) == 1
        assert len(app_file_handlers) == 1
    finally:
        for handler in batch_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
        for handler in app_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
        batch_logger.handlers = original_batch_handlers
        app_logger.handlers = original_app_handlers


def test_run_batch_passes_sequential_position_index(monkeypatch, tmp_path: Path):
    from app.config.models import AppSettings
    from scripts import batch_uci_runner as runner

    captured_positions: list[int | None] = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "configure_logging", lambda: None)
    monkeypatch.setattr(runner, "load_settings", lambda: AppSettings())
    monkeypatch.setattr(runner, "build_metadata_store", lambda settings: None)
    monkeypatch.setattr(runner, "_setup_batch_logger", lambda log_path: None)

    def _fake_run_single_dataset(
        uci_id: int,
        dataset_name: str,
        target: str,
        task_type: str,
        store,
        batch_id: str,
        total_datasets: int = 100,
        position_index: int | None = None,
    ) -> BatchRunItemRecord:
        captured_positions.append(position_index)
        return BatchRunItemRecord(
            item_id=f"{batch_id}::{uci_id}::{target}",
            batch_id=batch_id,
            uci_id=uci_id,
            dataset_name=dataset_name,
            target_column=target,
            task_type=task_type,
            status=BatchItemStatus.SUCCESS,
        )

    monkeypatch.setattr(runner, "run_single_dataset", _fake_run_single_dataset)

    runner.run_batch(
        [
            (53, "Iris", "class", "classification"),
            (109, "Wine", "class", "classification"),
        ],
        batch_id="batch-test",
    )

    assert captured_positions == [1, 2]


def test_full_resume_updates_batch_record_counts(monkeypatch, tmp_path: Path):
    """When ALL datasets are already completed, batch record counts must still be set."""
    from app.config.models import AppSettings
    from scripts import batch_uci_runner as runner

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "configure_logging", lambda: None)
    monkeypatch.setattr(runner, "load_settings", lambda: AppSettings())
    monkeypatch.setattr(runner, "_setup_batch_logger", lambda log_path: None)

    batch_id = "batch-full-resume"
    already_done = [
        BatchRunItemRecord(
            item_id=f"{batch_id}::53::class",
            batch_id=batch_id,
            uci_id=53,
            dataset_name="Iris",
            target_column="class",
            task_type="classification",
            status=BatchItemStatus.SUCCESS,
            metadata={"declared_target": "class"},
        ),
        BatchRunItemRecord(
            item_id=f"{batch_id}::109::class",
            batch_id=batch_id,
            uci_id=109,
            dataset_name="Wine",
            target_column="class",
            task_type="classification",
            status=BatchItemStatus.SUCCESS,
            metadata={"declared_target": "class"},
        ),
    ]

    upserted_runs: list = []

    class FakeStore:
        def list_batch_items(self, bid, limit=500):
            return already_done

        def upsert_batch_run(self, record):
            upserted_runs.append(record.model_copy(deep=True))

        def upsert_batch_item(self, item):
            pass

    monkeypatch.setattr(runner, "build_metadata_store", lambda settings: FakeStore())

    runner.run_batch(
        [
            (53, "Iris", "class", "classification"),
            (109, "Wine", "class", "classification"),
        ],
        batch_id=batch_id,
        resume=True,
    )

    final = upserted_runs[-1]
    assert final.completed_count == 2, f"Expected 2, got {final.completed_count}"
    assert final.status.value == "completed"


def test_known_case_sensitive_target_mappings():
    """Regression: targets that previously had wrong casing must match their UCI column names."""
    from scripts import batch_uci_runner as runner

    targets_by_uci_id = {uci_id: target for uci_id, _name, target, _task in runner.UCI_DATASETS}

    # Mammographic Mass: column is "Severity" not "severity"
    assert targets_by_uci_id[161] == "Severity"
    # Spambase: column is "Class" not "class"
    assert targets_by_uci_id[94] == "Class"
    # Heart Failure Clinical Records: column is "death_event" not "DEATH_EVENT"
    assert targets_by_uci_id[519] == "death_event"
    # ILPD: column is "Selector" not "is_patient"
    assert targets_by_uci_id[225] == "Selector"