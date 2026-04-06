"""Excel workbook loader."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from app.ingestion.base import BaseLoader
from app.ingestion.errors import ParseFailureError, UnsupportedSourceError
from app.ingestion.schemas import DatasetInputSpec
from app.ingestion.types import IngestionSourceType
from app.ingestion.url_loader import fetch_url_bytes


class ExcelLoader(BaseLoader):
    """Load local or remote Excel workbooks."""

    supported_source_types = (IngestionSourceType.EXCEL,)

    def load_raw_dataframe(
        self,
        input_spec: DatasetInputSpec,
        *,
        row_limit: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        try:
            if input_spec.path is not None:
                path = Path(input_spec.path)
                if not path.exists():
                    raise UnsupportedSourceError(f"Local Excel file not found: {path}")
                excel_source: str | BytesIO = str(path)
                content_type = None
                final_url = None
            elif input_spec.url is not None:
                payload, content_type, final_url = fetch_url_bytes(input_spec.url)
                excel_source = BytesIO(payload)
            else:
                raise UnsupportedSourceError("Excel ingestion requires either a local path or a URL.")

            with pd.ExcelFile(excel_source) as excel_file:
                sheet_names = list(excel_file.sheet_names)
                sheet_name = input_spec.excel_sheet if input_spec.excel_sheet is not None else 0
                dataframe = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=row_limit)
                actual_sheet_name = self._resolve_sheet_name(sheet_name, sheet_names)

            source_details = {
                "source_kind": "url" if input_spec.url is not None else "path",
                "available_sheet_names": sheet_names,
                "sheet_name": actual_sheet_name,
                "content_type": content_type,
                "final_url": final_url,
            }
            return dataframe, source_details
        except UnsupportedSourceError:
            raise
        except IndexError as exc:
            raise ParseFailureError("The requested Excel sheet index is out of range.") from exc
        except ValueError as exc:
            raise ParseFailureError(f"Failed to load the Excel workbook: {exc}") from exc
        except ImportError as exc:
            raise ParseFailureError(
                "Excel support requires the appropriate engine dependencies. Install 'openpyxl' and 'xlrd'."
            ) from exc

    def _resolve_sheet_name(self, sheet_name: str | int, sheet_names: list[str]) -> str | int:
        if isinstance(sheet_name, int):
            return sheet_names[sheet_name]
        return sheet_name
