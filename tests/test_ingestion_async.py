"""Async ingestion tests covering the non-blocking loader entrypoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest
import respx

from app.ingestion.factory import load_dataset_async, preview_dataset_async
from app.ingestion.schemas import DatasetInputSpec
from app.ingestion.types import IngestionSourceType
from app.ingestion.uci_loader import list_available_uci_datasets_async


class TestAsyncIngestionFactory:
    @pytest.mark.asyncio
    async def test_preview_local_csv_async(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text("feature,target\n1,0\n2,1\n", encoding="utf-8")

        loaded = await preview_dataset_async(
            DatasetInputSpec(source_type=IngestionSourceType.CSV, path=csv_path),
            rows=1,
        )

        assert loaded.dataframe.shape == (1, 2)
        assert loaded.metadata.applied_row_limit == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_load_remote_csv_async(self) -> None:
        url = "https://example.com/data.csv"
        csv_body = b"x,y\n1,2\n3,4\n"
        respx.head(url).mock(return_value=httpx.Response(200, headers={"content-type": "text/csv"}))
        respx.get(url).mock(return_value=httpx.Response(200, content=csv_body, headers={"content-type": "text/csv"}))

        loaded = await load_dataset_async(
            DatasetInputSpec(source_type=IngestionSourceType.URL_FILE, url=url)
        )

        assert loaded.dataframe.shape == (2, 2)
        assert loaded.metadata.source_details["routed_source_type"] == "csv"
        assert loaded.metadata.source_details["load_strategy"] == "streamed_temp_file_chunked_read_csv_async"

    @pytest.mark.asyncio
    @respx.mock
    async def test_load_html_table_from_url_async(self) -> None:
        url = "https://example.com/table"
        html = """
        <html><body>
        <table>
            <thead><tr><th>name</th><th>score</th></tr></thead>
            <tbody><tr><td>a</td><td>10</td></tr><tr><td>b</td><td>20</td></tr></tbody>
        </table>
        </body></html>
        """
        respx.get(url).mock(return_value=httpx.Response(200, text=html, headers={"content-type": "text/html"}))

        loaded = await load_dataset_async(
            DatasetInputSpec(source_type=IngestionSourceType.HTML_TABLE, url=url)
        )

        assert loaded.dataframe.shape == (2, 2)
        assert loaded.metadata.source_details["load_strategy"] == "streamed_temp_file_bounded_html_parse_async"

    @pytest.mark.asyncio
    @respx.mock
    async def test_url_file_sniff_fallback_async(self) -> None:
        url = "https://example.com/download?id=42"
        html = "<html><body><table><tr><th>a</th></tr><tr><td>1</td></tr></table></body></html>"
        respx.head(url).mock(return_value=httpx.Response(405))
        respx.get(url).mock(return_value=httpx.Response(200, text=html, headers={"content-type": "text/plain"}))

        loaded = await load_dataset_async(
            DatasetInputSpec(source_type=IngestionSourceType.URL_FILE, url=url)
        )

        assert loaded.dataframe.iloc[0, 0] == 1
        assert loaded.metadata.source_details["probe_method"] == "get-sniff"
        assert loaded.metadata.source_details["routed_source_type"] == "html_table"

    @pytest.mark.asyncio
    @respx.mock
    async def test_load_remote_excel_async(self) -> None:
        url = "https://example.com/dataset.xlsx"
        buffer = pd.io.common.BytesIO()
        with pd.ExcelWriter(buffer) as writer:
            pd.DataFrame({"age": [10, 20], "target": [0, 1]}).to_excel(writer, index=False)
        payload = buffer.getvalue()

        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=payload,
                headers={
                    "content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                },
            )
        )

        loaded = await load_dataset_async(
            DatasetInputSpec(source_type=IngestionSourceType.EXCEL, url=url)
        )

        assert loaded.dataframe.shape == (2, 2)
        assert loaded.metadata.source_details["load_strategy"] in {
            "openpyxl_read_only_async",
            "pandas_read_excel_async",
        }


class TestAsyncUCIHelpers:
    @pytest.mark.asyncio
    async def test_load_uci_dataset_async(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ucimlrepo = pytest.importorskip("ucimlrepo")

        mock_ds = MagicMock()
        mock_ds.data.ids = pd.DataFrame()
        mock_ds.data.features = pd.DataFrame({"feature": [1, 2]})
        mock_ds.data.targets = pd.DataFrame({"target": [0, 1]})
        mock_ds.data.original = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
        mock_ds.data.headers = ["feature", "target"]
        mock_ds.metadata.get = lambda key, default=None: {
            "uci_id": 1,
            "name": "Demo",
            "abstract": "demo",
            "area": "testing",
            "task": "Classification",
            "characteristics": [],
            "num_instances": 2,
            "num_features": 1,
            "feature_types": ["Integer"],
            "target_col": ["target"],
            "index_col": [],
            "has_missing_values": "no",
            "missing_values_symbol": None,
            "year_of_dataset_creation": 2020,
            "dataset_doi": None,
            "creators": [],
            "intro_paper": None,
            "repository_url": None,
            "data_url": None,
            "external_url": None,
            "additional_info": None,
        }.get(key, default)
        mock_ds.metadata.name = "Demo"
        mock_ds.metadata.uci_id = 1
        mock_ds.variables = pd.DataFrame({"name": ["feature", "target"]})

        monkeypatch.setattr(ucimlrepo, "fetch_ucirepo", MagicMock(return_value=mock_ds))

        loaded = await load_dataset_async(
            DatasetInputSpec(source_type=IngestionSourceType.UCI_REPO, uci_id=1)
        )

        assert loaded.dataframe.shape == (2, 2)
        assert loaded.metadata.source_details["source_kind"] == "uci_repo"

    @pytest.mark.asyncio
    async def test_list_available_uci_datasets_async(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ucimlrepo = pytest.importorskip("ucimlrepo")

        def _fake_list_available_datasets(*, filter=None, search=None, area=None) -> None:
            print("Dataset Name    ID")
            print("Iris           53")
            print("Wine           109")

        monkeypatch.setattr(ucimlrepo, "list_available_datasets", _fake_list_available_datasets)

        rows = await list_available_uci_datasets_async(search="demo")

        assert rows == [
            {"uci_id": 53, "name": "Iris"},
            {"uci_id": 109, "name": "Wine"},
        ]
