# Release Notes Template

Use this template when drafting a manual GitHub release or portfolio handoff summary.

This repository does not generate release notes automatically.

## Title

`AutoTabML Studio vX.Y.Z`

## Summary

One short paragraph covering what changed and why it matters for local users or maintainers.

## Highlights

- local workflow improvements:
- packaging or maintainer changes:
- documentation or demo updates:

## Verification

```bash
autotabml --version
autotabml info
pytest -q
```

## Known Limits To Mention

- local-first workflow only
- notebook mode is still a placeholder
- `colab_mcp` is scaffolded only
- optional extras are still required for profiling, benchmark, experiment, and MLflow-backed workflows

## Assets

- link only to real screenshots or demo media that actually exist in `docs/assets/`