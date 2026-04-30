"""Security & correctness tests for the notebook generator."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat
import pytest

from app.notebooks.generator import (
    NotebookGenerationError,
    generate_job_notebook,
)


def _read_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return nbformat.read(fh, as_version=4)


def _all_code_sources(nb: dict) -> list[str]:
    return [cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code"]


def _string_constants(source: str) -> list[str]:
    """Return all string-literal constants in ``source``.

    Used to assert that hostile substrings appear only as inert data, never
    as parsed identifiers / calls / statements.
    """

    tree = ast.parse(source)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _non_string_source(source: str) -> str:
    """Return ``source`` with all string-literal contents blanked out.

    Any hostile token that survives in the result is genuinely executable
    code, not a quoted constant.
    """

    tree = ast.parse(source)
    spans: list[tuple[int, int]] = []
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def _abs(lineno: int, col: int) -> int:
        return offsets[lineno - 1] + col

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.end_lineno is None or node.end_col_offset is None:
                continue
            spans.append(
                (
                    _abs(node.lineno, node.col_offset),
                    _abs(node.end_lineno, node.end_col_offset),
                )
            )
    spans.sort()
    out: list[str] = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            continue
        out.append(source[cursor:start])
        cursor = end
    out.append(source[cursor:])
    return "".join(out)


# ── Validation ─────────────────────────────────────────────────────────


class TestInputValidation:
    def test_rejects_unknown_job_type(self, tmp_path):
        with pytest.raises(NotebookGenerationError):
            generate_job_notebook(
                dataset_name="ds",
                job_type="malicious",
                output_dir=tmp_path,
            )

    def test_rejects_unknown_task_type(self, tmp_path):
        with pytest.raises(NotebookGenerationError):
            generate_job_notebook(
                dataset_name="ds",
                job_type="benchmark",
                task_type="rm -rf /",
                output_dir=tmp_path,
            )

    def test_rejects_path_with_nul_byte(self, tmp_path):
        with pytest.raises(NotebookGenerationError):
            generate_job_notebook(
                dataset_name="ds",
                job_type="benchmark",
                artifact_path="evil\x00.csv",
                output_dir=tmp_path,
            )

    def test_filename_is_sanitized(self, tmp_path):
        path = generate_job_notebook(
            dataset_name="../../etc/passwd",
            job_type="benchmark",
            output_dir=tmp_path,
        )
        # Resolved path stays inside output_dir
        assert path.resolve().is_relative_to(tmp_path.resolve())
        # No path separators leaked into the filename
        assert "/" not in path.name and "\\" not in path.name


# ── Injection resistance ───────────────────────────────────────────────


class TestInjectionResistance:
    def test_dataset_name_with_quotes_does_not_break_code(self, tmp_path):
        # A name designed to break naive f-string concatenation.
        hostile = "x'); import os; os.system('pwned'); #"
        path = generate_job_notebook(
            dataset_name=hostile,
            job_type="benchmark",
            target_column="target",
            output_dir=tmp_path,
        )
        nb = _read_notebook(path)
        for src in _all_code_sources(nb):
            # Every code cell must be syntactically valid Python.
            ast.parse(src)
            # Hostile tokens may exist inside string literals (safe), but
            # must NEVER appear as executable code.
            stripped = _non_string_source(src)
            assert "import os" not in stripped
            assert "os.system" not in stripped

    def test_target_column_with_newlines_is_neutralized(self, tmp_path):
        hostile_target = "target\n__import__('os').system('rm -rf /')"
        path = generate_job_notebook(
            dataset_name="ds",
            job_type="flaml",
            task_type="classification",
            target_column=hostile_target,
            output_dir=tmp_path,
        )
        nb = _read_notebook(path)
        for src in _all_code_sources(nb):
            stripped = _non_string_source(src)
            assert "__import__" not in stripped
            assert "rm -rf" not in stripped

    def test_metadata_with_hostile_keys_is_json_quoted(self, tmp_path):
        hostile_meta = {
            "best_estimator": "lgbm'); raise SystemExit('boom",
            "raw\nkey": "value\"with\"quotes",
        }
        path = generate_job_notebook(
            dataset_name="ds",
            job_type="flaml",
            metadata=hostile_meta,
            output_dir=tmp_path,
        )
        nb = _read_notebook(path)
        # Find the metadata code cell and confirm it parses & loads back correctly.
        meta_cells = [
            src for src in _all_code_sources(nb) if src.startswith("metadata = json.loads(")
        ]
        assert len(meta_cells) == 1
        ast.parse(meta_cells[0])
        # Extract the inner JSON-encoded literal and round-trip it.
        first_line = meta_cells[0].splitlines()[0]
        assert first_line.startswith("metadata = json.loads(")
        inner_str_literal = first_line[len("metadata = json.loads(") : -1]
        outer_decoded = json.loads(inner_str_literal)
        round_trip = json.loads(outer_decoded)
        assert round_trip == hostile_meta

    def test_artifact_path_with_quotes_is_json_quoted(self, tmp_path):
        hostile_path = str(tmp_path / "evil'); print('pwned'); #.csv")
        path = generate_job_notebook(
            dataset_name="ds",
            job_type="benchmark",
            artifact_path=hostile_path,
            output_dir=tmp_path,
        )
        nb = _read_notebook(path)
        for src in _all_code_sources(nb):
            stripped = _non_string_source(src)
            assert "pwned" not in stripped
        # The hostile path must round-trip exactly as a string constant.
        leaderboard_cells = [
            src for src in _all_code_sources(nb) if "leaderboard_path = Path(" in src
        ]
        assert leaderboard_cells
        constants = _string_constants(leaderboard_cells[0])
        assert any(hostile_path in c for c in constants)


# ── nbformat validity ──────────────────────────────────────────────────


class TestNotebookFormat:
    def test_generated_notebook_validates_against_nbformat(self, tmp_path):
        path = generate_job_notebook(
            dataset_name="iris",
            job_type="experiment",
            task_type="classification",
            target_column="species",
            metadata={"selected_model_name": "rf"},
            artifact_path=str(tmp_path / "model.pkl"),
            output_dir=tmp_path,
        )
        nb = _read_notebook(path)
        # Should not raise.
        nbformat.validate(nb)
        assert nb["nbformat"] == 4
        assert any(cell["cell_type"] == "code" for cell in nb["cells"])

    def test_cells_include_explicit_language_metadata(self, tmp_path):
        path = generate_job_notebook(
            dataset_name="iris",
            job_type="benchmark",
            output_dir=tmp_path,
        )
        nb = _read_notebook(path)
        for cell in nb["cells"]:
            expected = "markdown" if cell["cell_type"] == "markdown" else "python"
            assert cell["metadata"]["language"] == expected

    def test_all_code_cells_parse_as_python(self, tmp_path):
        for job_type in ("benchmark", "experiment", "flaml", "profiling", "validation"):
            path = generate_job_notebook(
                dataset_name=f"ds_{job_type}",
                job_type=job_type,
                task_type="classification",
                target_column="y",
                metadata={"k": "v"},
                artifact_path=str(tmp_path / f"a_{job_type}.csv"),
                summary_path=str(tmp_path / f"s_{job_type}.json"),
                output_dir=tmp_path,
            )
            nb = _read_notebook(path)
            for src in _all_code_sources(nb):
                ast.parse(src)
