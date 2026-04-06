"""Batch UCI dataset runner – runs validate → profile → benchmark for 100 UCI datasets.

Usage:
    python scripts/batch_uci_runner.py [--resume <batch_id>] [--start <index>] [--count <n>]

Tracks every step in the local SQLite database (batch_runs / batch_run_items tables)
and logs all output to artifacts/batch_runs/<batch_id>/batch.log.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import load_settings
from app.ingestion import DatasetInputSpec, IngestionSourceType, load_dataset
from app.logging_config import configure_logging
from app.storage import build_metadata_store, ensure_dataset_record
from app.storage.models import (
    BatchItemStatus,
    BatchRunItemRecord,
    BatchRunRecord,
    BatchRunStatus,
)

logger = logging.getLogger("batch_runner")

# ---------------------------------------------------------------------------
# 100 UCI datasets: (uci_id, dataset_name, target_column, task_hint)
# task_hint: "classification" | "regression" | "auto"
# ---------------------------------------------------------------------------
UCI_DATASETS: list[tuple[int, str, str, str]] = [
    # Classic small / medium classification
    (53, "Iris", "class", "classification"),
    (109, "Wine", "class", "classification"),
    (17, "Breast Cancer Wisconsin (Diagnostic)", "Diagnosis", "classification"),
    (15, "Breast Cancer Wisconsin (Original)", "Class", "classification"),
    (42, "Glass Identification", "Type_of_glass", "classification"),
    (111, "Zoo", "type", "classification"),
    (101, "Tic-Tac-Toe Endgame", "class", "classification"),
    (151, "Connectionist Bench (Sonar)", "class", "classification"),
    (52, "Ionosphere", "Class", "classification"),
    (267, "Banknote Authentication", "class", "classification"),
    (176, "Blood Transfusion Service Center", "Donated_Blood", "classification"),
    (39, "Ecoli", "class", "classification"),
    (80, "Optical Recognition of Handwritten Digits", "class", "classification"),
    (81, "Pen-Based Recognition of Handwritten Digits", "Class", "classification"),
    (602, "Dry Bean", "Class", "classification"),
    (545, "Rice (Cammeo and Osmancik)", "Class", "classification"),
    (850, "Raisin", "Class", "classification"),
    (451, "Breast Cancer Coimbra", "Classification", "classification"),
    (73, "Mushroom", "poisonous", "classification"),
    (19, "Car Evaluation", "class", "classification"),
    (76, "Nursery", "class", "classification"),
    (2, "Adult", "income", "classification"),
    (20, "Census Income", "income", "classification"),
    (222, "Bank Marketing", "y", "classification"),
    (94, "Spambase", "Class", "classification"),
    (327, "Phishing Websites", "result", "classification"),
    (159, "MAGIC Gamma Telescope", "class", "classification"),
    (22, "Chess (King-Rook vs. King-Pawn)", "wtoeg", "classification"),
    (105, "Congressional Voting Records", "Class", "classification"),
    (45, "Heart Disease", "num", "classification"),
    (145, "Statlog (Heart)", "heart-disease", "classification"),
    (519, "Heart Failure Clinical Records", "death_event", "classification"),
    (529, "Early Stage Diabetes Risk Prediction", "class", "classification"),
    (336, "Chronic Kidney Disease", "class", "classification"),
    (468, "Online Shoppers Purchasing Intention", "Revenue", "classification"),
    (225, "ILPD (Indian Liver Patient Dataset)", "Selector", "classification"),
    (14, "Breast Cancer", "Class", "classification"),
    (33, "Dermatology", "class", "classification"),
    (43, "Haberman's Survival", "survival_status", "classification"),
    (46, "Hepatitis", "Class", "classification"),
    (110, "Yeast", "localization_site", "classification"),
    (198, "Steel Plates Faults", "Pastry", "classification"),
    (12, "Balance Scale", "class", "classification"),
    (193, "Cardiotocography", "NSP", "classification"),
    (264, "EEG Eye State", "eyeDetection", "classification"),
    (212, "Vertebral Column", "class", "classification"),
    (329, "Diabetic Retinopathy Debrecen", "Class", "classification"),
    (863, "Maternal Health Risk", "RiskLevel", "classification"),
    (544, "Estimation of Obesity Levels", "NObeyesdad", "classification"),
    (571, "HCV data", "Category", "classification"),
    (95, "SPECT Heart", "OVERALL_DIAGNOSIS", "classification"),
    (563, "Iranian Churn", "Churn", "classification"),
    (601, "AI4I 2020 Predictive Maintenance", "Machine failure", "classification"),
    (697, "Predict Students Dropout and Academic Success", "Target", "classification"),
    (372, "HTRU2", "class", "classification"),
    (848, "Secondary Mushroom", "class", "classification"),
    (161, "Mammographic Mass", "Severity", "classification"),
    (713, "Auction Verification", "verification.result", "classification"),
    (878, "Cirrhosis Patient Survival Prediction", "Status", "classification"),
    (857, "Risk Factor Prediction of CKD", "class", "classification"),
    # Classification – larger / multiclass
    (59, "Letter Recognition", "lettr", "classification"),
    (146, "Statlog (Landsat Satellite)", "class", "classification"),
    (149, "Statlog (Vehicle Silhouettes)", "Class", "classification"),
    (342, "Mice Protein Expression", "class", "classification"),
    (78, "Page Blocks Classification", "class", "classification"),
    (350, "Default of Credit Card Clients", "default payment next month", "classification"),
    (471, "Electrical Grid Stability", "stabf", "classification"),
    (890, "AIDS Clinical Trials Group Study 175", "cid", "classification"),
    (603, "In-Vehicle Coupon Recommendation", "Y", "classification"),
    (856, "Higher Education Students Performance", "GRADE", "classification"),
    # Regression
    (1, "Abalone", "Rings", "regression"),
    (9, "Auto MPG", "mpg", "regression"),
    (10, "Automobile", "price", "regression"),
    (162, "Forest Fires", "area", "regression"),
    (165, "Concrete Compressive Strength", "csMPa", "regression"),
    (242, "Energy Efficiency", "Y1", "regression"),
    (291, "Airfoil Self-Noise", "ScaledSoundPressure", "regression"),
    (294, "Combined Cycle Power Plant", "PE", "regression"),
    (477, "Real Estate Valuation", "Y house price of unit area", "regression"),
    (560, "Seoul Bike Sharing Demand", "Rented Bike Count", "regression"),
    (186, "Wine Quality", "quality", "regression"),
    (275, "Bike Sharing", "cnt", "regression"),
    (320, "Student Performance", "G3", "regression"),
    (374, "Appliances Energy Prediction", "Appliances", "regression"),
    (189, "Parkinsons Telemonitoring", "total_UPDRS", "regression"),
    (464, "Superconductivty Data", "critical_temp", "regression"),
    (87, "Servo", "class", "regression"),
    (247, "ISTANBUL STOCK EXCHANGE", "ISE", "regression"),
    (332, "Online News Popularity", "shares", "regression"),
    (409, "Daily Demand Forecasting Orders", "Target (Total orders)", "regression"),
    (849, "Power Consumption of Tetouan City", "Zone 1 Power Consumption", "regression"),
    (851, "Steel Industry Energy Consumption", "Usage_kWh", "regression"),
    (597, "Productivity Prediction of Garment Employees", "actual_productivity", "regression"),
    (492, "Metro Interstate Traffic Volume", "traffic_volume", "regression"),
    (551, "Gas Turbine CO and NOx Emission", "CO", "regression"),
    (29, "Computer Hardware", "PRP", "regression"),
    (174, "Parkinsons", "status", "classification"),
    (27, "Credit Approval", "A16", "classification"),
    (184, "Acute Inflammations", "d1", "classification"),
    (244, "Fertility", "Output", "classification"),
    (277, "Thoracic Surgery Data", "Risk1Yr", "classification"),
]

SAMPLE_THRESHOLD = 10_000  # datasets > 10k rows get sampled


def _setup_batch_logger(log_path: Path) -> None:
    """Add a file handler to the batch_runner logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _attach_file_handler(target_logger: logging.Logger) -> None:
        resolved_path = str(log_path.resolve())
        for existing in target_logger.handlers:
            if isinstance(existing, logging.FileHandler) and Path(existing.baseFilename).resolve() == Path(resolved_path):
                return

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        target_logger.addHandler(file_handler)

    _attach_file_handler(logger)
    _attach_file_handler(logging.getLogger("app"))


def _declared_target_from_item(item: BatchRunItemRecord) -> str:
    """Return the stable declared target used to identify a dataset variant across resumes."""

    declared_target = item.metadata.get("declared_target")
    if declared_target is not None:
        return str(declared_target)

    prefix = f"{item.batch_id}::{item.uci_id}::"
    if item.item_id.startswith(prefix):
        return item.item_id[len(prefix):]
    return ""


def _build_resume_state(
    datasets: list[tuple[int, str, str, str]],
    existing_items: list[BatchRunItemRecord],
) -> tuple[dict[int, list[BatchRunItemRecord]], set[tuple[int, str]], Counter[int], int, int]:
    """Return the normalized resume lookup and persisted counters for a batch."""

    items_by_uci: dict[int, list[BatchRunItemRecord]] = {}
    completed_keys: set[tuple[int, str]] = set()
    variant_counts = Counter(uci_id for uci_id, _, _, _ in datasets)
    success_count = 0
    skipped_count = 0

    for item in existing_items:
        items_by_uci.setdefault(item.uci_id, []).append(item)
        if item.status == BatchItemStatus.SUCCESS:
            success_count += 1
        elif item.status == BatchItemStatus.SKIPPED:
            skipped_count += 1

        if item.status in (BatchItemStatus.SUCCESS, BatchItemStatus.SKIPPED):
            completed_keys.add((item.uci_id, _declared_target_from_item(item)))

    return items_by_uci, completed_keys, variant_counts, success_count, skipped_count


def _should_skip_completed_dataset(
    uci_id: int,
    declared_target: str,
    *,
    items_by_uci: dict[int, list[BatchRunItemRecord]],
    completed_keys: set[tuple[int, str]],
    variant_counts: Counter[int],
) -> bool:
    """Return True when the current dataset variant has already completed in this batch."""

    if (uci_id, declared_target or "") in completed_keys:
        return True

    # Backward-compatible fallback for older batches that tracked only one item per UCI id.
    if variant_counts.get(uci_id, 0) == 1:
        return any(
            item.status in (BatchItemStatus.SUCCESS, BatchItemStatus.SKIPPED)
            for item in items_by_uci.get(uci_id, [])
        )

    return False


def _run_validate(locator: str, target: str | None) -> dict:
    """Run validation via CLI-equivalent programmatic call."""
    from app.validation.schemas import ValidationRuleConfig
    from app.validation.service import validate_dataset

    settings = load_settings()
    metadata_store = build_metadata_store(settings)

    spec = DatasetInputSpec(
        source_type=IngestionSourceType.UCI_REPO,
        uci_id=int(locator),
    )
    loaded = load_dataset(spec)
    name = loaded.metadata.display_name or f"uci_{locator}"
    if metadata_store:
        ensure_dataset_record(metadata_store, loaded, dataset_name=name)

    config = ValidationRuleConfig(
        target_column=target,
        min_row_count=settings.validation.min_row_threshold,
        null_warn_pct=settings.validation.null_warn_pct,
        null_fail_pct=settings.validation.null_fail_pct,
    )
    summary, bundle = validate_dataset(
        loaded.dataframe,
        config,
        dataset_name=name,
        artifacts_dir=settings.validation.artifacts_dir,
        loaded_dataset=loaded,
        metadata_store=metadata_store,
    )
    return {
        "status": "failed" if summary.has_failures else "success",
        "passed": summary.passed_count,
        "warnings": summary.warning_count,
        "failures": summary.failed_count,
        "rows": summary.row_count,
        "columns": summary.column_count,
        "loaded": loaded,
        "name": name,
    }


def _run_profile(loaded, name: str) -> dict:
    """Run profiling via programmatic call."""
    from app.profiling.ydata_runner import is_ydata_available
    from app.profiling.service import profile_dataset

    settings = load_settings()
    metadata_store = build_metadata_store(settings)

    if not is_ydata_available():
        return {"status": "skipped", "reason": "ydata-profiling not installed"}

    summary, bundle = profile_dataset(
        loaded.dataframe,
        mode=settings.profiling.default_mode,
        dataset_name=name,
        artifacts_dir=settings.profiling.artifacts_dir,
        loaded_dataset=loaded,
        metadata_store=metadata_store,
        large_dataset_row_threshold=settings.profiling.large_dataset_row_threshold,
        large_dataset_col_threshold=settings.profiling.large_dataset_col_threshold,
        sampling_row_threshold=settings.profiling.sampling_row_threshold,
        sample_size=settings.profiling.sample_size,
    )
    return {
        "status": "success",
        "rows": summary.row_count,
        "columns": summary.column_count,
        "missing_pct": summary.missing_cells_pct,
    }


def _run_benchmark(loaded, name: str, target: str, task_type: str) -> dict:
    """Run benchmark via programmatic call."""
    from app.modeling.benchmark.errors import BenchmarkError
    from app.modeling.benchmark.schemas import BenchmarkConfig, BenchmarkSplitConfig, BenchmarkTaskType
    from app.modeling.benchmark.service import benchmark_dataset

    settings = load_settings()
    metadata_store = build_metadata_store(settings)

    sample_rows = None
    if len(loaded.dataframe) > SAMPLE_THRESHOLD:
        sample_rows = SAMPLE_THRESHOLD

    config = BenchmarkConfig(
        target_column=target,
        task_type=BenchmarkTaskType(task_type),
        prefer_gpu=settings.benchmark.prefer_gpu,
        split=BenchmarkSplitConfig(
            test_size=settings.benchmark.default_test_size,
            random_state=settings.benchmark.default_random_state,
        ),
        sample_rows=sample_rows,
        top_k=settings.benchmark.ui_default_top_k,
        timeout_seconds=settings.benchmark.timeout_seconds,
    )
    dataset_fingerprint = loaded.metadata.content_hash or loaded.metadata.schema_hash

    bundle = benchmark_dataset(
        loaded.dataframe,
        config,
        dataset_name=name,
        dataset_fingerprint=dataset_fingerprint,
        loaded_dataset=loaded,
        metadata_store=metadata_store,
        execution_backend=settings.execution.backend,
        workspace_mode=settings.workspace_mode,
        artifacts_dir=settings.benchmark.artifacts_dir,
        classification_default_metric=settings.benchmark.default_classification_ranking_metric,
        regression_default_metric=settings.benchmark.default_regression_ranking_metric,
        sampling_row_threshold=settings.benchmark.sampling_row_threshold,
        suggested_sample_rows=settings.benchmark.suggested_sample_rows,
        mlflow_experiment_name=settings.benchmark.mlflow_experiment_name,
        tracking_uri=settings.tracking.tracking_uri,
        registry_uri=settings.tracking.registry_uri,
    )
    return {
        "status": "success",
        "best_model": bundle.summary.best_model_name,
        "best_score": bundle.summary.best_score,
        "ranking_metric": bundle.summary.ranking_metric,
        "mlflow_run_id": bundle.mlflow_run_id,
        "duration": bundle.summary.benchmark_duration_seconds,
        "task_type": bundle.summary.task_type.value,
        "models_evaluated": len(bundle.top_models),
    }


def _detect_target_and_task(loaded, declared_target: str, declared_task: str) -> tuple[str, str]:
    """Validate target column exists; fall back to UCI metadata target_columns if needed."""
    df = loaded.dataframe
    source_details = loaded.metadata.source_details or {}

    # Check declared target
    if declared_target in df.columns:
        return declared_target, declared_task

    # Try UCI metadata target_columns
    uci_targets = source_details.get("target_columns", [])
    if uci_targets:
        for t in uci_targets:
            if t in df.columns:
                logger.warning(
                    "Declared target '%s' not in columns; falling back to UCI target '%s'",
                    declared_target, t,
                )
                return t, declared_task

    # Last resort: use the last column
    fallback = df.columns[-1]
    logger.warning(
        "No known target column found; using last column '%s'",
        fallback,
    )
    return str(fallback), "auto"


def run_single_dataset(
    uci_id: int,
    dataset_name: str,
    target: str,
    task_type: str,
    store,
    batch_id: str,
    total_datasets: int = 100,
    position_index: int | None = None,
) -> BatchRunItemRecord:
    """Execute the full pipeline for one UCI dataset and return its item record."""
    now = datetime.now(timezone.utc)
    # Include target in item_id to allow same uci_id with different targets
    item_suffix = f"{uci_id}::{target}" if target else str(uci_id)
    item = BatchRunItemRecord(
        item_id=f"{batch_id}::{item_suffix}",
        batch_id=batch_id,
        uci_id=uci_id,
        dataset_name=dataset_name,
        target_column=target,
        task_type=task_type,
        status=BatchItemStatus.RUNNING,
        created_at=now,
        updated_at=now,
    )
    item.metadata["declared_target"] = target
    item.metadata["resolved_target"] = target
    if store:
        store.upsert_batch_item(item)

    start_time = time.monotonic()
    logger.info("=" * 70)
    logger.info("START [%d/%d] uci:%d  %s  target=%s  task=%s",
                position_index or uci_id, total_datasets, uci_id, dataset_name, target, task_type)

    try:
        # Step 1: Validate
        logger.info("[%s] Step 1/3: Validating...", dataset_name)
        val_result = _run_validate(str(uci_id), target)
        item.validation_status = val_result["status"]
        item.row_count = val_result.get("rows")
        item.column_count = val_result.get("columns")
        loaded = val_result["loaded"]
        name = val_result["name"]

        # Detect actual target/task after loading
        actual_target, actual_task = _detect_target_and_task(loaded, target, task_type)
        item.target_column = actual_target
        item.task_type = actual_task
        item.metadata["resolved_target"] = actual_target
        item.metadata["target_fallback_used"] = actual_target != target
        logger.info("[%s] Validation %s (rows=%s, cols=%s)",
                     dataset_name, val_result["status"], item.row_count, item.column_count)

        # Step 2: Profile
        logger.info("[%s] Step 2/3: Profiling...", dataset_name)
        try:
            prof_result = _run_profile(loaded, name)
            item.profiling_status = prof_result["status"]
            logger.info("[%s] Profiling %s", dataset_name, prof_result["status"])
        except Exception as prof_exc:
            item.profiling_status = "failed"
            logger.warning("[%s] Profiling failed (non-fatal): %s", dataset_name, prof_exc)

        # Step 3: Benchmark
        logger.info("[%s] Step 3/3: Benchmarking (target=%s, task=%s)...",
                     dataset_name, actual_target, actual_task)
        bench_result = _run_benchmark(loaded, name, actual_target, actual_task)
        item.benchmark_status = bench_result["status"]
        item.best_model = bench_result.get("best_model")
        item.best_score = bench_result.get("best_score")
        item.ranking_metric = bench_result.get("ranking_metric")
        item.mlflow_run_id = bench_result.get("mlflow_run_id")
        item.task_type = bench_result.get("task_type", actual_task)
        item.status = BatchItemStatus.SUCCESS
        logger.info("[%s] Benchmark SUCCESS: best=%s score=%.4f metric=%s",
                     dataset_name, item.best_model, item.best_score or 0, item.ranking_metric)

    except Exception as exc:
        item.status = BatchItemStatus.FAILED
        item.error_message = str(exc)[:500]
        logger.error("[%s] FAILED: %s", dataset_name, exc)
        logger.debug(traceback.format_exc())

    elapsed = time.monotonic() - start_time
    item.duration_seconds = round(elapsed, 2)
    item.updated_at = datetime.now(timezone.utc)

    if store:
        store.upsert_batch_item(item)

    logger.info("END   [%s] status=%s  duration=%.1fs", dataset_name, item.status.value, elapsed)
    return item


def run_batch(
    datasets: list[tuple[int, str, str, str]],
    *,
    batch_id: str | None = None,
    resume: bool = False,
) -> None:
    """Execute the full batch run for the given dataset list."""
    configure_logging()
    settings = load_settings()
    store = build_metadata_store(settings)

    if batch_id is None:
        batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Setup batch log file
    batch_dir = Path("artifacts/batch_runs") / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    _setup_batch_logger(batch_dir / "batch.log")

    logger.info("=" * 70)
    logger.info("BATCH RUN: %s  datasets=%d", batch_id, len(datasets))
    logger.info("=" * 70)

    # Determine already-completed items for resume
    items_by_uci: dict[int, list[BatchRunItemRecord]] = {}
    completed_keys: set[tuple[int, str]] = set()
    dataset_variant_counts = Counter(uci_id for uci_id, _, _, _ in datasets)
    if resume and store:
        existing_items = store.list_batch_items(batch_id, limit=max(len(datasets), 500))
        items_by_uci, completed_keys, dataset_variant_counts, success_count, skip_count = _build_resume_state(
            datasets,
            existing_items,
        )
        logger.info("Resume mode: %d datasets already completed, skipping", len(completed_keys))
    else:
        success_count = 0
        skip_count = 0

    # Create or update batch run record
    now = datetime.now(timezone.utc)
    batch_record = BatchRunRecord(
        batch_id=batch_id,
        batch_name=f"UCI {len(datasets)}-Dataset Batch Run",
        total_datasets=len(datasets),
        status=BatchRunStatus.RUNNING,
        started_at=now,
        updated_at=now,
    )
    if store:
        store.upsert_batch_run(batch_record)

    fail_count = 0

    for idx, (uci_id, name, target, task) in enumerate(datasets, 1):
        if _should_skip_completed_dataset(
            uci_id,
            target,
            items_by_uci=items_by_uci,
            completed_keys=completed_keys,
            variant_counts=dataset_variant_counts,
        ):
            logger.info("SKIP [%d/%d] uci:%d %s target=%s (already completed)", idx, len(datasets), uci_id, name, target)
            continue

        item = run_single_dataset(
            uci_id,
            name,
            target,
            task,
            store,
            batch_id,
            total_datasets=len(datasets),
            position_index=idx,
        )

        if item.status == BatchItemStatus.SUCCESS:
            success_count += 1
            completed_keys.add((item.uci_id, _declared_target_from_item(item)))
            items_by_uci.setdefault(item.uci_id, []).append(item)
        elif item.status == BatchItemStatus.FAILED:
            fail_count += 1
        else:
            skip_count += 1
            completed_keys.add((item.uci_id, _declared_target_from_item(item)))
            items_by_uci.setdefault(item.uci_id, []).append(item)

        # Update batch record
        batch_record.completed_count = success_count
        batch_record.failed_count = fail_count
        batch_record.skipped_count = skip_count
        batch_record.updated_at = datetime.now(timezone.utc)
        if store:
            store.upsert_batch_run(batch_record)

        done_count = success_count + fail_count + skip_count
        logger.info(
            "PROGRESS: %d/%d done  (success=%d, failed=%d, skipped=%d)",
            done_count, len(datasets), success_count, fail_count, skip_count,
        )

    # Finalize – ensure counts reflect the complete session (covers full-resume edge case)
    batch_record.completed_count = success_count
    batch_record.failed_count = fail_count
    batch_record.skipped_count = skip_count
    if fail_count == 0:
        batch_record.status = BatchRunStatus.COMPLETED
    elif success_count > 0:
        batch_record.status = BatchRunStatus.PARTIAL
    else:
        batch_record.status = BatchRunStatus.FAILED
    batch_record.updated_at = datetime.now(timezone.utc)
    if store:
        store.upsert_batch_run(batch_record)

    # Write summary JSON
    summary = {
        "batch_id": batch_id,
        "total": len(datasets),
        "success": success_count,
        "failed": fail_count,
        "skipped": skip_count,
        "status": batch_record.status.value,
        "started_at": batch_record.started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = batch_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("BATCH COMPLETE: %s", batch_record.status.value)
    logger.info("  Success: %d  Failed: %d  Skipped: %d  Total: %d",
                success_count, fail_count, skip_count, len(datasets))
    logger.info("  Summary: %s", summary_path)
    logger.info("  Log: %s", batch_dir / "batch.log")
    logger.info("=" * 70)

    # Print to console
    print(f"\n{'=' * 60}")
    print(f"BATCH RUN COMPLETE: {batch_record.status.value}")
    print(f"  Batch ID: {batch_id}")
    print(f"  Success: {success_count}  Failed: {fail_count}  Skipped: {skip_count}")
    print(f"  Summary: {summary_path}")
    print(f"  Log: {batch_dir / 'batch.log'}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch UCI dataset runner")
    parser.add_argument("--resume", default=None, help="Resume a previous batch run by batch_id")
    parser.add_argument("--start", type=int, default=0, help="Start index in the dataset list (0-based)")
    parser.add_argument("--count", type=int, default=100, help="Number of datasets to run")
    args = parser.parse_args()

    datasets = UCI_DATASETS[args.start : args.start + args.count]

    if args.resume:
        run_batch(datasets, batch_id=args.resume, resume=True)
    else:
        run_batch(datasets)


if __name__ == "__main__":
    main()
