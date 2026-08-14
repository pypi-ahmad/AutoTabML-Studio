"""Portable prediction bundle generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import zipfile

from pydantic import BaseModel


class DeploymentBundle(BaseModel):
    archive_path: Path
    model_name: str
    sha256: str


def export_deployment_bundle(
    *,
    model_path: Path,
    metadata_path: Path,
    output_path: Path,
    provenance_path: Path | None = None,
) -> DeploymentBundle:
    """Create API, Docker, and command-line deployment assets for one trusted model."""

    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Model and metadata files are required.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Model metadata is not valid JSON: {exc}") from exc
    if metadata.get("research_only") is True or metadata.get("deployable") is False:
        raise ValueError("research-only model contexts cannot be exported for deployment.")
    output_path = output_path.with_suffix(".zip")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "autotabml-deployment"
        model_dir = root / "model"
        model_dir.mkdir(parents=True)
        shutil.copy2(model_path, model_dir / model_path.name)
        shutil.copy2(metadata_path, model_dir / metadata_path.name)
        if provenance_path and provenance_path.is_file():
            shutil.copy2(provenance_path, model_dir / provenance_path.name)
        (root / "serve.py").write_text(_SERVER_TEMPLATE, encoding="utf-8")
        (root / "predict.py").write_text(
            _CLI_TEMPLATE.replace("__MODEL_FILE__", model_path.name).replace("__METADATA_FILE__", metadata_path.name),
            encoding="utf-8",
        )
        (root / "Dockerfile").write_text(_DOCKERFILE, encoding="utf-8")
        wheel = _build_current_wheel(root / "wheels")
        package_requirement = (
            f"./wheels/{wheel.name}[benchmark,experiment,flaml,serve]"
            if wheel is not None
            else "autotabml-studio[benchmark,experiment,flaml,serve]>=0.3.0"
        )
        (root / "requirements.txt").write_text(f"{package_requirement}\n", encoding="utf-8")
        (root / "README.md").write_text(_README, encoding="utf-8")
        manifest = {path.relative_to(root).as_posix(): _sha256(path) for path in root.rglob("*") if path.is_file()}
        (root / "checksums.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(root))
    return DeploymentBundle(archive_path=output_path, model_name=model_path.stem, sha256=_sha256(output_path))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_current_wheel(output_dir: Path) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(  # nosec B603
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(output_dir), "."],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        shutil.rmtree(output_dir, ignore_errors=True)
        return None
    return next(output_dir.glob("*.whl"), None)


_SERVER_TEMPLATE = '''"""Generated AutoTabML prediction API."""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AutoTabML Model")

class PredictionPayload(BaseModel):
    records: list[dict]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metadata")
def metadata():
    return {"model_directory": "model"}

@app.post("/predict")
def predict(payload: PredictionPayload):
    if len(payload.records) > 10000:
        raise ValueError("Maximum batch size is 10,000 rows.")
    # Load through AutoTabML's checksum-verified prediction service.
    from predict import run_prediction
    return {"predictions": run_prediction(payload.records)}
'''

_CLI_TEMPLATE = '''"""Generated standalone prediction entry point."""
import argparse
import json
from pathlib import Path

import pandas as pd

from app.prediction import BatchPredictionRequest, ModelSourceType, PredictionService

def run_prediction(records):
    root = Path(__file__).resolve().parent
    service = PredictionService(
        artifacts_dir=root / "outputs",
        history_path=root / "outputs" / "history.jsonl",
        local_model_dirs=[root / "model"],
        local_metadata_dirs=[root / "model"],
        registry_enabled=False,
    )
    result = service.predict_batch(
        BatchPredictionRequest(
            source_type=ModelSourceType.LOCAL_SAVED_MODEL,
            model_path=root / "model" / "__MODEL_FILE__",
            metadata_path=root / "model" / "__METADATA_FILE__",
            dataframe=pd.DataFrame(records),
        )
    )
    return result.scored_dataframe.to_dict(orient="records")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    args = parser.parse_args()
    print(json.dumps(run_prediction(json.loads(Path(args.input_json).read_text(encoding="utf-8"))), default=str))
'''

_DOCKERFILE = """FROM python:3.12-slim
RUN useradd --create-home appuser
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
USER appuser
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
"""

_README = """# AutoTabML deployment bundle

This bundle contains a checksum-backed model and generated API/CLI scaffolding.
Bind locally by default. Add authentication and TLS before internet exposure.
"""

__all__ = ["DeploymentBundle", "export_deployment_bundle"]
