"""High-level validation service – the main entry point for validation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from app.ingestion.schemas import LoadedDataset
from app.storage import AppMetadataStore, record_validation_job
from app.validation.artifacts import write_artifacts
from app.validation.base import BaseValidationService
from app.validation.gx_context import is_gx_available
from app.validation.gx_runner import run_gx_validation
from app.validation.rules import run_app_rules
from app.validation.schemas import (
    ValidationArtifactBundle,
    ValidationResultSummary,
    ValidationRuleConfig,
)
from app.validation.summary import build_summary

logger = logging.getLogger(__name__)


class GreatExpectationsValidationService(BaseValidationService):
    """Validation service combining app-level rules and GX expectations."""

    def validate(
        self,
        df: pd.DataFrame,
        config: ValidationRuleConfig,
        *,
        dataset_name: str | None = None,
    ) -> ValidationResultSummary:
        gx_enabled = is_gx_available()
        gx_checks = run_gx_validation(df, config) if gx_enabled else []
        gx_failed = any(
            check.check_name == "gx_execution" and not check.passed
            for check in gx_checks
        )
        checks = run_app_rules(
            df,
            config,
            gx_managed_rules=gx_enabled and not gx_failed,
        )
        checks.extend(gx_checks)
        return build_summary(checks, df, dataset_name=dataset_name)


def validate_dataset(
    df: pd.DataFrame,
    config: ValidationRuleConfig | None = None,
    *,
    dataset_name: str | None = None,
    artifacts_dir: Path | None = None,
    loaded_dataset: LoadedDataset | None = None,
    metadata_store: AppMetadataStore | None = None,
) -> tuple[ValidationResultSummary, ValidationArtifactBundle | None]:
    """Convenience function: validate and optionally write artifacts.

    Returns (summary, artifact_bundle_or_None).
    """
    if config is None:
        config = ValidationRuleConfig()

    service = GreatExpectationsValidationService()
    summary = service.validate(df, config, dataset_name=dataset_name)

    bundle: ValidationArtifactBundle | None = None
    if artifacts_dir is not None:
        bundle = write_artifacts(summary, artifacts_dir)

    if metadata_store is not None:
        record_validation_job(
            metadata_store,
            summary,
            artifacts=bundle,
            loaded_dataset=loaded_dataset,
        )

    return summary, bundle
