from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from app.deployment import export_deployment_bundle
from app.modeling.foundation import TabFMLoadedContext
from app.modeling.foundation.tracking import log_foundation_run
from app.prediction import ModelSourceType, PredictionRequest, PredictionTaskType
from app.prediction.loader import LocalTabFMContextLoader
from app.storage.models import AppJobType
from app.tracking.schemas import RunType


class _ContextService:
    def load_context(self, metadata_path, **kwargs):  # noqa: ANN001, ANN003
        return TabFMLoadedContext(
            native_model=object(),
            task_type="classification",
            feature_columns=["a", "b"],
            target_column="target",
            metadata={"research_only": True, "deployable": False},
        )


def _write_context_metadata(directory: Path) -> Path:
    metadata = directory / "demo_tabfm_metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "trusted_source": "autotabml_tabfm_research_context_v1",
                "artifact_format": "tabfm_context_v1",
                "task_type": "classification",
                "target_column": "target",
                "feature_columns": ["a", "b"],
                "context_file": "demo_tabfm_context.json",
                "research_only": True,
                "deployable": False,
            }
        ),
        encoding="utf-8",
    )
    return metadata


def test_foundation_job_and_run_types_are_first_class():
    assert AppJobType.TABFM.value == "tabfm"
    assert AppJobType.TIMESFM.value == "timesfm"
    assert RunType.TABFM.value == "tabfm"
    assert RunType.TIMESFM.value == "timesfm"


def test_tabfm_context_loader_discovers_and_normalizes_context(tmp_path: Path):
    metadata = _write_context_metadata(tmp_path)
    loader = LocalTabFMContextLoader(metadata_dirs=[tmp_path], service=_ContextService())

    references = loader.discover()
    loaded = loader.load(
        PredictionRequest(
            source_type=ModelSourceType.LOCAL_SAVED_MODEL,
            model_identifier=references[0].model_identifier,
            metadata_path=metadata,
        )
    )

    assert references[0].task_type == PredictionTaskType.CLASSIFICATION
    assert loaded.loader_name == "LocalTabFMContextLoader"
    assert loaded.scorer_kind == "sklearn_like"
    assert loaded.feature_columns == ["a", "b"]
    assert loaded.metadata["deployable"] is False


def test_deployment_export_rejects_research_only_tabfm_context(tmp_path: Path):
    context = tmp_path / "context.json"
    context.write_text("{}", encoding="utf-8")
    metadata = _write_context_metadata(tmp_path)

    with pytest.raises(ValueError, match="research-only"):
        export_deployment_bundle(
            model_path=context,
            metadata_path=metadata,
            output_path=tmp_path / "bundle.zip",
        )


def test_cli_parses_tabfm_and_timesfm_commands(monkeypatch):
    from app import cli as cli_module

    observed: list[tuple[str, object]] = []
    monkeypatch.setattr(cli_module, "cmd_tabfm_run", lambda args: observed.append(("tabfm", args)))
    monkeypatch.setattr(
        "sys.argv",
        [
            "autotabml",
            "tabfm-run",
            "data.csv",
            "--target",
            "label",
            "--allow-download",
            "--accept-tabfm-license",
        ],
    )
    cli_module.main()
    assert observed[0][0] == "tabfm"
    assert observed[0][1].target == "label"
    assert observed[0][1].accept_tabfm_license is True

    monkeypatch.setattr(cli_module, "cmd_timesfm_forecast", lambda args: observed.append(("timesfm", args)))
    monkeypatch.setattr(
        "sys.argv",
        [
            "autotabml",
            "timesfm-forecast",
            "data.csv",
            "--timestamp",
            "date",
            "--target",
            "sales",
            "--group",
            "store",
        ],
    )
    cli_module.main()
    assert observed[1][0] == "timesfm"
    assert observed[1][1].group == "store"


def test_foundation_tracking_logs_only_summary_artifact(monkeypatch, tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    context = tmp_path / "private_context.json"
    context.write_text("private", encoding="utf-8")
    logged: dict[str, object] = {}

    @contextmanager
    def start_run(**kwargs):  # noqa: ANN003, ANN202
        logged["run"] = kwargs
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda value: logged.update(tracking_uri=value),
        set_experiment=lambda value: logged.update(experiment=value),
        start_run=start_run,
        set_tags=lambda value: logged.update(tags=value),
        log_params=lambda value: logged.update(params=value),
        log_metrics=lambda value: logged.update(metrics=value),
        log_artifact=lambda value: logged.update(artifact=value),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    run_id, warning = log_foundation_run(
        run_type="tabfm",
        dataset_name="private-dataset",
        params={"context_rows": 128},
        metrics={"accuracy": 0.9},
        summary_path=summary,
    )

    assert run_id == "run-1"
    assert warning is None
    assert logged["artifact"] == str(summary)
    assert str(context) not in str(logged)
