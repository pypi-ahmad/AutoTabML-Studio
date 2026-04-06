# Release / Demo Checklist

## Before A Demo Or Public Release

- run `pytest`
- verify `autotabml --version`
- verify `autotabml info`
- run `python -m app.release_metadata`
- run `autotabml init-local-storage`
- run `autotabml doctor`
- verify optional extras needed for the demo are installed
- verify MLflow-backed pages only if MLflow is installed and configured locally
- verify the sample dataset you plan to use fits the chosen benchmark/experiment path

## Documentation Checks

- update `CHANGELOG.md`
- draft the release summary from [release-notes-template.md](release-notes-template.md)
- README links resolve
- docs links resolve
- command examples match `autotabml --help`
- screenshot sections remain clearly marked as placeholders until assets exist

## Demo Asset Checks

- capture real screenshots into `docs/assets/screenshots/`
- add a real social preview image into `docs/assets/social-preview/` before claiming it exists publicly
- keep filenames aligned with the asset plan in [assets/README.md](assets/README.md)

## Pre-Public Repo Checks

- confirm `.env` is ignored and not committed
- confirm README does not claim notebook execution or remote execution are complete
- confirm `python -m app.release_metadata` still passes with the committed public license and maintainer/contact metadata, and update those fields if the public owner/contact details change before publication
- confirm repo description/topics/social preview are set in GitHub settings if publishing publicly
