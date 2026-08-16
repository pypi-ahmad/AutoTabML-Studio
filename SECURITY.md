# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.4.x   | ✅ Yes    |
| 0.3.x   | ⚠️ Best-effort (legacy) |
| < 0.3   | ❌ No     |

The current release line is **0.4.x**. Security fixes land on `main`
and are released as patch or minor versions as the severity warrants.

## Security Posture — v0.4.0

The v0.2.0 release established the hardened foundation; v0.3.0 extended it, and
v0.4.0 adds consent-gated local foundation models. The following controls are
active by default.

### Ingestion

- **SSRF-resistant HTTP client** (`app.security.safe_http`): every
  remote URL is checked against a scheme allowlist (`http`/`https`),
  has its hostname resolved through `socket.getaddrinfo` and
  every returned IP checked against loopback, link-local, private,
  multicast, reserved, and unspecified ranges. Redirects are
  followed manually with a hop cap and re-validated. Responses
  are streamed and aborted at a configurable byte cap.
- **CSV / Excel formula-injection protection** (`app.security.safe_csv`):
  any cell or column whose value starts with `=`, `+`, `-`, or
  `@` is prefixed with a single quote before CSV export, so a
  malicious file cannot trigger formula execution when the export
  is opened in Excel / LibreOffice / Google Sheets.
- **Trusted local-artifact loader** (`app.security.trusted_artifacts`):
  any pickle / skops model file is loaded only after (a) its
  resolved path is verified to live under a configured trust
  root, (b) the SHA256 sidecar is read and verified, and
  (c) the loader rejects path-traversal attempts.

### Secrets

- **Secret masking** in logs: the JSON formatter scrubs
  `sk-...` / `AI...` / `Bearer ...` / `user:password@host` patterns
  before the log line is emitted.
- **API keys are not persisted** to `~/.autotabml/settings.json` —
  the file is for non-secret preferences only. The Settings page
  keeps API keys in `st.session_state` and reads them from
  environment variables.
- **No outbound telemetry by default**: `~/.streamlit/config.toml`
  sets `gatherUsageStats = false` at both the client and browser
  levels so the "local-first" claim is verifiable.

### Dependencies

- **Locked** via `uv.lock`; CI runs `uv lock --check` on every
  push to prevent drift.
- **Pinned** for the GitHub Actions supply chain: all third-party
  Actions are pinned to commit SHAs (see `.github/workflows/*.yml`).
- **Scanned** on every push via the `security.yml` workflow using
  `detect-secrets`, `gitleaks`, `bandit`, and `pip-audit`.

### Foundation Models

- **Explicit download consent:** TabFM and TimesFM checkpoints are pinned to exact
  Hugging Face revisions and are not downloaded until the user opts in.
- **License gate:** TabFM additionally requires acceptance of its
  `tabfm-non-commercial-v1.0` weights license. Saved TabFM contexts are marked
  research-only and are rejected by registry and deployment-export workflows.
- **Local data boundary:** foundation-model inference remains local. MLflow receives
  aggregate parameters and metrics, never raw rows, sampled contexts, predictions,
  or forecast tables.

### Build & Distribution

- **PEP 517 isolated build** via hatchling; the build runs in a
  fresh venv so no developer's environment can leak into the
  sdist / wheel.
- **`twine check dist/*`** is run as part of `release-readiness.yml`
  on every tag push; the workflow fails if the metadata is
  invalid.
- The wheel and sdist produced by `uv run --no-sync python -m build`
  ship with the SPDX license identifier, a populated
  `py.typed` marker (PEP 561), and the full source tree.

## Reporting a Vulnerability

Please report potential vulnerabilities through **GitHub Security
Advisories** at:

  <https://github.com/pypi-ahmad/AutoTabML-Studio/security/advisories/new>

When reporting, include:

- A clear description of the issue.
- An impact assessment (who is affected, what data is at risk).
- Reproduction steps or a proof of concept.
- A suggested remediation, if you have one in mind.

**Do not** post exploit details in a public GitHub Issue — use the
private advisory channel. The maintainer will triage and respond
within **7 days**. Critical issues will receive a patch release as
soon as the fix is verified.

## Response Process

1. **Triage** — the maintainer confirms reproduction, assesses
   severity (CVSS-style), and assigns a target version.
2. **Patch** — a fix is developed on a private branch.
3. **Verify** — the fix passes the full test suite, the CI coverage gate, and the
   security scan.
4. **Disclose** — once a patch is released, the advisory is
   published with full impact, fix version, and mitigation
   notes.

## Hardening Guide

If you are deploying AutoTabML Studio in a shared or hostile
environment, the following additional steps are recommended:

- Run the Streamlit UI behind a reverse proxy (nginx, Caddy)
  that enforces TLS, rate-limiting, and request size limits.
- Pin the container image by digest: `docker run
  ghcr.io/pypi-ahmad/autotabml-studio@sha256:...`.
- Drop the `--server.address=0.0.0.0` flag in the Dockerfile's
  CMD and bind to a Unix socket or a sidecar network.
- Enable filesystem sandboxing (e.g. bubblewrap, containerd
  with `read_only=true`) so the local artifact directory cannot
  escape its mount.
- Rotate provider API keys regularly and consider using
  short-lived credentials from your provider's identity service
  (e.g. OpenAI's service accounts) instead of long-lived keys.

## Data Responsibility

AutoTabML Studio runs entirely on your machine. The maintainer has no access to
your data and accepts no responsibility for data you process with this software.
You are responsible for compliance with any applicable regulations. See
[DISCLAIMER.md](DISCLAIMER.md) for the full statement.

## Acknowledgements

We thank the security researchers and contributors who have
reported issues to the project. Responsible disclosure is what
keeps the workbench safe for everyone.
