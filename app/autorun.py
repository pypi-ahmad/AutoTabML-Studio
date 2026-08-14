"""Guided, engine-aware Auto Run planning and execution."""

from __future__ import annotations

from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from app.drift import build_drift_baseline
from app.evaluation import EvaluationReport, evaluate_model
from app.explainability import ModelExplanation, explain_global
from app.modeling.flaml.schemas import FlamlConfig, FlamlSearchConfig, FlamlTaskType
from app.modeling.flaml.service import FlamlAutoMLService
from app.modeling.flaml.setup_runner import is_flaml_available
from app.provenance import ProvenanceManifest, build_provenance, write_provenance


class AutoRunMode(str, Enum):
    AUTO = "auto"
    QUICK = "quick"
    BALANCED = "balanced"
    DEEP = "deep"


class AutoRunConfig(BaseModel):
    target_column: str
    task_type: str = "auto"
    mode: AutoRunMode = AutoRunMode.AUTO
    time_budget: int = Field(default=120, ge=10)
    random_seed: int = 42
    model_name: str = "autorun-model"


class AutoRunPlan(BaseModel):
    target_column: str
    task_type: str
    engine: str
    primary_metric: str
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AutoRunResult(BaseModel):
    plan: AutoRunPlan
    model_path: Path
    metadata_path: Path
    evaluation: EvaluationReport
    explanation: ModelExplanation
    provenance: ProvenanceManifest
    artifact_paths: dict[str, Path]


def suggest_targets(dataframe: pd.DataFrame) -> list[str]:
    preferred = ["target", "label", "class", "outcome", "y"]
    by_name = [column for wanted in preferred for column in dataframe.columns if str(column).lower() == wanted]
    candidates = [
        str(column)
        for column in dataframe.columns
        if column not in by_name and not str(column).lower().endswith(("id", "_id"))
    ]
    return [*by_name, *candidates]


def plan_auto_run(dataframe: pd.DataFrame, config: AutoRunConfig) -> AutoRunPlan:
    if config.target_column not in dataframe.columns:
        raise ValueError(f"Target column '{config.target_column}' was not found.")
    target = dataframe[config.target_column].dropna()
    if target.nunique() < 2:
        raise ValueError("Target must contain at least two distinct values.")
    task = config.task_type
    if task == "auto":
        unique = target.nunique()
        class_limit = min(20, max(2, int(len(target) * 0.1)))
        task = "classification" if not pd.api.types.is_numeric_dtype(target) or unique <= class_limit else "regression"
    reasons = ["FLAML provides a time-bounded search and a directly savable model."]
    warnings: list[str] = []
    if config.mode == AutoRunMode.QUICK:
        warnings.append("Quick mode uses the same FLAML engine with a shorter search budget.")
    if not is_flaml_available():
        warnings.append("Install the 'flaml' extra before launching Auto Run.")
    return AutoRunPlan(
        target_column=config.target_column,
        task_type=task,
        engine="flaml",
        primary_metric="balanced_accuracy" if task == "classification" else "r2",
        reasons=reasons,
        warnings=warnings,
    )


def run_auto_run(
    dataframe: pd.DataFrame,
    config: AutoRunConfig,
    *,
    artifacts_dir: Path,
    models_dir: Path,
    job_id: str | None = None,
) -> AutoRunResult:
    """Run FLAML on a training split and evaluate on an untouched holdout."""

    from sklearn.model_selection import train_test_split

    plan = plan_auto_run(dataframe, config)
    if not is_flaml_available():
        raise RuntimeError("Auto Run requires the FLAML optional dependency.")
    train, holdout = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=config.random_seed,
        stratify=dataframe[config.target_column] if plan.task_type == "classification" else None,
    )
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(dataframe, index=True).values.tobytes()
    ).hexdigest()
    service = FlamlAutoMLService(artifacts_dir=artifacts_dir, models_dir=models_dir)
    budget = 30 if config.mode == AutoRunMode.QUICK else config.time_budget
    bundle = service.run_automl(
        train,
        FlamlConfig(
            target_column=config.target_column,
            task_type=FlamlTaskType(plan.task_type),
            search=FlamlSearchConfig(time_budget=budget, seed=config.random_seed),
        ),
        dataset_name="Auto Run dataset",
        dataset_fingerprint=fingerprint,
    )
    bundle = service.save_best_model(bundle, save_name=config.model_name)
    metadata = bundle.saved_model_metadata
    if metadata is None:
        raise RuntimeError("FLAML did not produce a saved model.")
    metadata_path = (
        bundle.artifacts.saved_model_metadata_path
        if bundle.artifacts is not None and bundle.artifacts.saved_model_metadata_path is not None
        else metadata.model_path.with_name(
            f"{metadata.model_path.stem}_flaml_saved_model_metadata_{metadata.model_path.stem}.json"
        )
    )
    if not metadata_path.is_file():
        raise RuntimeError("FLAML model metadata was not persisted.")
    model = bundle.runtime.automl_instance
    features = [column for column in dataframe.columns if column != config.target_column]
    evaluation = evaluate_model(model, holdout[features], holdout[config.target_column], task_type=plan.task_type)
    explanation = explain_global(model, holdout[features], holdout[config.target_column])
    job_dir = artifacts_dir / "autorun" / (job_id or fingerprint[:12])
    job_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = _write_json(job_dir / "evaluation.json", evaluation.model_dump(mode="json"))
    explanation_path = _write_json(job_dir / "explanation.json", explanation.model_dump(mode="json"))
    baseline = build_drift_baseline(train[features])
    baseline_path = _write_json(
        job_dir / "drift-baseline.json",
        baseline.model_dump(mode="json"),
    )
    provenance = build_provenance(
        engine="flaml",
        task_type=plan.task_type,
        target_column=config.target_column,
        dataset_fingerprint=fingerprint,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        random_seed=config.random_seed,
        configuration=config.model_dump(mode="json"),
        job_id=job_id,
        model_path=metadata.model_path,
        repo_root=Path.cwd(),
    )
    provenance_path = write_provenance(provenance, job_dir / "provenance.json")
    model_baseline_path = _write_json(
        metadata.model_path.with_name(f"{metadata.model_path.stem}_drift_baseline.json"),
        baseline.model_dump(mode="json"),
    )
    model_provenance_path = write_provenance(
        provenance,
        metadata.model_path.with_name(f"{metadata.model_path.stem}_provenance.json"),
    )
    return AutoRunResult(
        plan=plan,
        model_path=metadata.model_path,
        metadata_path=metadata_path,
        evaluation=evaluation,
        explanation=explanation,
        provenance=provenance,
        artifact_paths={
            "evaluation": evaluation_path,
            "explanation": explanation_path,
            "drift_baseline": baseline_path,
            "provenance": provenance_path,
            "model_drift_baseline": model_baseline_path,
            "model_provenance": model_provenance_path,
        },
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)
    return path


__all__ = [
    "AutoRunConfig",
    "AutoRunMode",
    "AutoRunPlan",
    "AutoRunResult",
    "plan_auto_run",
    "run_auto_run",
    "suggest_targets",
]
