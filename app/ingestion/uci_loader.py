"""UCI ML Repository dataset loader via the ``ucimlrepo`` package."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import re
from typing import Any

import pandas as pd

from app.errors import log_and_wrap
from app.ingestion.base import BaseLoader
from app.ingestion.errors import IngestionError, RemoteAccessError
from app.ingestion.schemas import DatasetInputSpec
from app.ingestion.types import IngestionSourceType

logger = logging.getLogger(__name__)


def list_available_uci_datasets(
    *,
    filter: str | None = None,
    search: str | None = None,
    area: str | None = None,
) -> list[dict[str, Any]]:
    """Return structured UCI catalog rows by parsing ``list_available_datasets`` output."""

    ucimlrepo = _import_ucimlrepo()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ucimlrepo.list_available_datasets(filter=filter, search=search, area=area)
    return _parse_catalog_output(buffer.getvalue())


async def list_available_uci_datasets_async(
    *,
    filter: str | None = None,
    search: str | None = None,
    area: str | None = None,
) -> list[dict[str, Any]]:
    """Async counterpart of :func:`list_available_uci_datasets`."""

    return await asyncio.to_thread(
        list_available_uci_datasets,
        filter=filter,
        search=search,
        area=area,
    )


def _import_ucimlrepo():
    try:
        import ucimlrepo
    except ImportError as exc:
        raise RemoteAccessError(
            "UCI Repository ingestion requires the 'ucimlrepo' package. "
            "Install it with: pip install 'ucimlrepo>=0.0.7'"
        ) from exc
    return ucimlrepo


def _parse_catalog_output(output: str) -> list[dict[str, Any]]:
    if not output.strip() or output.strip() == "No datasets found":
        return []

    rows: list[dict[str, Any]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-"):
            continue
        if line.startswith("The following datasets are available"):
            continue
        if line == "Dataset Name    ID":
            continue
        if line == "Dataset Name ID":
            continue
        if line == "Dataset Name\tID":
            continue

        parts = re.split(r"\s{2,}", line)
        if len(parts) < 2 or not parts[-1].isdigit():
            continue

        rows.append(
            {
                "uci_id": int(parts[-1]),
                "name": " ".join(parts[:-1]).strip(),
            }
        )
    return rows


def _to_dataframe(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, pd.Series):
        return value.to_frame()
    return pd.DataFrame(value)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, pd.Index):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


class UCIRepoLoader(BaseLoader):
    """Fetch a dataset from the UCI ML Repository by ID or name."""

    supported_source_types = (IngestionSourceType.UCI_REPO,)

    def load_raw_dataframe(
        self,
        input_spec: DatasetInputSpec,
        *,
        row_limit: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        ucimlrepo = _import_ucimlrepo()

        uci_id = input_spec.uci_id
        uci_name = input_spec.uci_name
        if uci_id is None and not uci_name:
            raise IngestionError(
                "UCI Repository ingestion requires a dataset ID (uci_id) or name (uci_name)."
            )
        if uci_id is not None and uci_name:
            raise IngestionError(
                "UCI Repository ingestion accepts either a dataset ID or a name, not both."
            )

        try:
            if uci_id is not None:
                dataset = ucimlrepo.fetch_ucirepo(id=uci_id)
            else:
                dataset = ucimlrepo.fetch_ucirepo(name=uci_name)
        except Exception as exc:  # noqa: BLE001 - external library may raise arbitrary errors
            identifier = f"id={uci_id}" if uci_id is not None else f"name='{uci_name}'"
            log_and_wrap(
                logger,
                exc,
                operation="ingestion.fetch_uci_dataset",
                wrap_with=RemoteAccessError,
                message=f"Failed to fetch UCI dataset ({identifier}): {exc}",
                context={"identifier": identifier},
            )

        ids = _to_dataframe(getattr(dataset.data, "ids", None))
        features = _to_dataframe(getattr(dataset.data, "features", None))
        targets = _to_dataframe(getattr(dataset.data, "targets", None))
        original = _to_dataframe(getattr(dataset.data, "original", None))
        headers = _to_builtin(getattr(dataset.data, "headers", None))

        if not original.empty:
            dataframe = original.copy(deep=True)
        else:
            frames = [frame for frame in (ids, features, targets) if not frame.empty]
            dataframe = pd.concat(frames, axis=1) if frames else pd.DataFrame()

        if row_limit is not None:
            dataframe = dataframe.head(row_limit).copy()

        meta = dataset.metadata
        additional_info = _to_builtin(meta.get("additional_info")) if hasattr(meta, "get") else None
        variables = _to_dataframe(getattr(dataset, "variables", None))
        source_details: dict[str, Any] = {
            "source_kind": "uci_repo",
            "uci_id": meta.get("uci_id"),
            "uci_name": meta.get("name"),
            "uci_abstract": meta.get("abstract"),
            "uci_area": meta.get("area"),
            "uci_task": meta.get("task"),
            "uci_characteristics": _to_builtin(meta.get("characteristics")),
            "uci_num_instances": meta.get("num_instances"),
            "uci_num_features": meta.get("num_features"),
            "uci_feature_types": _to_builtin(meta.get("feature_types")),
            "uci_target_col": _to_builtin(meta.get("target_col")),
            "uci_index_col": _to_builtin(meta.get("index_col")),
            "uci_has_missing_values": meta.get("has_missing_values"),
            "uci_missing_values_symbol": meta.get("missing_values_symbol"),
            "uci_year": meta.get("year_of_dataset_creation"),
            "uci_dataset_doi": meta.get("dataset_doi"),
            "uci_creators": _to_builtin(meta.get("creators")),
            "uci_intro_paper": _to_builtin(meta.get("intro_paper")),
            "uci_repository_url": meta.get("repository_url"),
            "uci_data_url": meta.get("data_url"),
            "uci_external_url": meta.get("external_url"),
            "uci_additional_info": additional_info,
            "uci_headers": headers or [str(column) for column in dataframe.columns.tolist()],
            "uci_original_shape": [int(value) for value in (original.shape if not original.empty else dataframe.shape)],
            "uci_loaded_shape": [int(value) for value in dataframe.shape],
            "uci_id_columns": [str(column) for column in ids.columns.tolist()],
            "feature_columns": features.columns.tolist(),
            "target_columns": targets.columns.tolist() if targets is not None else [],
            "id_columns": ids.columns.tolist(),
        }

        if not variables.empty:
            source_details["uci_variables"] = variables.to_dict(orient="records")
            source_details["uci_variable_columns"] = [str(column) for column in variables.columns.tolist()]

        logger.info(
            "UCI dataset '%s' (id=%s) loaded: %d rows × %d cols",
            meta.get("name"),
            meta.get("uci_id"),
            len(dataframe),
            len(dataframe.columns),
        )

        return dataframe, source_details
