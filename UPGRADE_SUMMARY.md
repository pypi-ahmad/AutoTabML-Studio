# UPGRADE_SUMMARY.md

> One-page upgrade cheat sheet for AutoTabML Studio 0.3.0.

## 0.2.x → 0.3.0

### Did anything break?

No. Version 0.3.0 is additive. The CLI, public service interfaces, SQLite and
MLflow layouts, artifact tree, and `~/.autotabml/settings.json` remain
compatible. No data migration is required.

### Upgrade

```bash
git pull
uv sync --locked --all-groups
uv run autotabml --version   # 0.3.0
uv run autotabml doctor
uv run pytest tests/ -q      # 729 selected; 30 optional tests deselected
```

Install only the optional extras you use. Current `main` declares `tabfm` and
`profiling` as incompatible because their upstream `typeguard` ranges do not
overlap; keep those two workflows in separate environments.

### What is new?

| Area | What landed |
| --- | --- |
| Auto Run | Guided background workflow with progress, cancellation, evaluation, explanations, provenance, and drift baselines |
| Delivery | FastAPI, standalone CLI, and non-root Docker deployment bundles |
| Desktop | Windows double-click launcher for the Streamlit UI |
| AI tooling | Six-tier hosted-model cost calculator with exact token arithmetic and pricing notes |
| Foundation models (`main`) | Revision-pinned TabFM research evaluation and TimesFM 2.5 forecasting with explicit download consent |
| Local UI (`main`) | Streamlit, Docker, and Compose now use `http://localhost:8561` |
| Performance | Lower-allocation CSV ingestion and simpler delimiter handling |
| Maintainability | Removed unused abstractions and prototypes; refreshed architecture and knowledge graphs |
| Security | Loopback-bound local services plus stronger URL, artifact, and log safeguards |

### Need help?

- Detailed release notes: `RELEASE_NOTES_v0.3.0.md`
- Full history: `CHANGELOG.md`
- Older 0.1.x → 0.2.0 migration: `MIGRATION_GUIDE.md`
- Security reporting and supported versions: `SECURITY.md`
