# UPGRADE_SUMMARY.md

> One-page upgrade cheat sheet for AutoTabML Studio 0.4.0.

## 0.3.x → 0.4.0

### Did anything break?

No. Version 0.4.0 is additive. The CLI, public service interfaces, SQLite and
MLflow layouts, artifact tree, and `~/.autotabml/settings.json` remain
compatible. No data migration is required.

### Upgrade

```bash
git pull
uv sync --locked --all-groups
uv run autotabml --version   # 0.4.0
uv run autotabml doctor
uv run pytest tests/ -q      # 729 selected; 30 optional tests deselected
```

Install only the optional extras you use. Version 0.4.0 declares `tabfm` and
`profiling` as incompatible because their upstream `typeguard` ranges do not
overlap; keep those two workflows in separate environments.

### What is new?

| Area | What landed |
| --- | --- |
| Foundation models | Revision-pinned TabFM research evaluation and TimesFM 2.5 forecasting with explicit download consent |
| TabFM safety | Non-commercial license gate, checksum-backed saved contexts, and registry/deployment blocking |
| Forecasting | Single/grouped TimesFM forecasts, q10–q90 uncertainty, and optional holdout backtests |
| Local UI | Streamlit, Docker, and Compose now use `http://localhost:8561` |
| Packaging | Docker and `make install` select compatible extras; TabFM remains in a separate environment |
| Security | Foundation-model inference stays local and MLflow receives aggregate-only summaries |

### Need help?

- Detailed release notes: `RELEASE_NOTES_v0.4.0.md`
- Full history: `CHANGELOG.md`
- Older release migrations: `MIGRATION_GUIDE.md`
- Security reporting and supported versions: `SECURITY.md`
