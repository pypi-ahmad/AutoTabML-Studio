"""Tests for public release metadata validation."""

from __future__ import annotations

from pathlib import Path

from app.release_metadata import check_public_release_metadata, validate_public_release_metadata


class TestValidatePublicReleaseMetadata:
    def test_reports_missing_license_and_contacts(self):
        issues = validate_public_release_metadata({"name": "autotabml-studio"})

        assert "[project].license is required before a public release." in issues
        assert (
            "At least one [project].authors or [project].maintainers entry is required before a public release."
            in issues
        )

    def test_accepts_license_and_named_contact_with_email(self):
        issues = validate_public_release_metadata(
            {
                "license": "MIT",
                "maintainers": [{"name": "AutoTabML Maintainer", "email": "maintainer@example.com"}],
            }
        )

        assert issues == []

    def test_reports_missing_contact_email_even_when_name_is_present(self):
        issues = validate_public_release_metadata(
            {
                "license": {"file": "LICENSE"},
                "authors": [{"name": "AutoTabML Maintainer"}],
            }
        )

        assert "At least one public maintainer/contact email is required before a public release." in issues


class TestCheckPublicReleaseMetadata:
    def test_reads_pyproject_from_disk(self, tmp_path: Path):
        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(
            """
[project]
name = "autotabml-studio"
license = "MIT"
maintainers = [{name = "AutoTabML Maintainer", email = "maintainer@example.com"}]
""".strip(),
            encoding="utf-8",
        )

        issues = check_public_release_metadata(pyproject_path)

        assert issues == []

    def test_committed_repo_pyproject_passes_public_release_metadata_check(self):
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"

        issues = check_public_release_metadata(pyproject_path)

        assert issues == []