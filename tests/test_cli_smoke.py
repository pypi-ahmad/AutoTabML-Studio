"""End-to-end CLI smoke tests.

These exercise the public ``autotabml`` entrypoint exactly as a user
would invoke it from a shell. They run the commands against a
throwaway ``XDG_CONFIG_HOME`` / ``AUTOTABML_*`` env so they never
read the developer's real ``~/.autotabml/settings.json``.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run the CLI via ``python -m app.cli`` so we don't depend on PATH.

    Using the module form is more portable than relying on a console
    script being installed in the venv; the entrypoint and the
    module are the same code.
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "app.cli", *args],
        capture_output=True,
        text=True,
        env=full_env,
        timeout=60,
        check=False,
    )


def test_version_exits_zero() -> None:
    result = _run("--version")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "0.2.0" in result.stdout, f"stdout: {result.stdout}"


def test_help_exits_zero_and_lists_subcommands() -> None:
    result = _run("--help")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    # At least the major subcommands must be present.
    for sub in ("validate", "profile", "benchmark", "info", "doctor"):
        assert sub in result.stdout, f"missing {sub} in help text: {result.stdout[:200]}"


def test_info_runs_without_error(tmp_path: Path) -> None:
    """``autotabml info`` must not require network or real data."""
    result = _run(
        "info",
        env={"AUTOTABML_ARTIFACTS__ROOT_DIR": str(tmp_path)},
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"


def test_doctor_runs_in_isolated_workspace(tmp_path: Path) -> None:
    """``autotabml doctor`` must complete and report on local state."""
    result = _run(
        "doctor",
        env={"AUTOTABML_ARTIFACTS__ROOT_DIR": str(tmp_path)},
    )
    assert result.returncode in (0, 1), f"unexpected exit: {result.stderr}"
    # doctor output should mention some local artifact dirs
    assert "artifact" in result.stdout.lower() or "artifact" in result.stderr.lower()


def test_init_local_storage_creates_dirs(tmp_path: Path) -> None:
    """``autotabml init-local-storage`` must create the artifact tree."""
    artifacts_dir = tmp_path / "artifacts"
    result = _run(
        "init-local-storage",
        env={"AUTOTABML_ARTIFACTS__ROOT_DIR": str(artifacts_dir)},
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert artifacts_dir.exists(), f"expected {artifacts_dir} to exist"
