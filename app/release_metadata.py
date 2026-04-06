"""Public release metadata checks for packaging and publishing."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised only on Python < 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


def load_project_metadata(pyproject_path: Path) -> dict[str, Any]:
    """Load the PEP 621 project table from pyproject.toml."""

    with pyproject_path.open("rb") as handle:
        payload = tomllib.load(handle)

    project = payload.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml is missing a [project] table.")
    return project


def validate_public_release_metadata(project_metadata: Mapping[str, Any]) -> list[str]:
    """Return blocking issues for a public package release."""

    issues: list[str] = []

    if not _has_license_metadata(project_metadata.get("license")):
        issues.append("[project].license is required before a public release.")

    contacts = _collect_contacts(project_metadata)
    if not contacts:
        issues.append("At least one [project].authors or [project].maintainers entry is required before a public release.")
        return issues

    if not any(_normalized(entry.get("name")) for entry in contacts):
        issues.append("At least one public maintainer/author name is required before a public release.")
    if not any(_normalized(entry.get("email")) for entry in contacts):
        issues.append("At least one public maintainer/contact email is required before a public release.")

    return issues


def check_public_release_metadata(pyproject_path: Path) -> list[str]:
    """Load and validate the project's public release metadata."""

    return validate_public_release_metadata(load_project_metadata(pyproject_path))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public release metadata check as a CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Validate required public release metadata in pyproject.toml.")
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pyproject.toml",
        help="Path to the pyproject.toml file to validate.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        issues = check_public_release_metadata(args.pyproject)
    except Exception as exc:
        print(f"Release metadata check failed to run: {exc}", file=sys.stderr)
        return 1

    if issues:
        print("Public release metadata check failed:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    print("Public release metadata check passed.")
    return 0


def _collect_contacts(project_metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    contacts: list[Mapping[str, Any]] = []
    for key in ("maintainers", "authors"):
        entries = project_metadata.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping):
                contacts.append(entry)
    return contacts


def _has_license_metadata(raw_license: Any) -> bool:
    if isinstance(raw_license, str):
        return bool(_normalized(raw_license))
    if isinstance(raw_license, Mapping):
        return bool(_normalized(raw_license.get("text")) or _normalized(raw_license.get("file")))
    return False


def _normalized(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())