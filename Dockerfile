# syntax=docker/dockerfile:1.7
#
# AutoTabML Studio — production-grade multi-stage image.
#
# Build:
#   docker build -t autotabml-studio:0.2.0 .
#
# Run (Streamlit UI):
#   docker run --rm -p 8501:8501 \
#     -v $(pwd)/artifacts:/app/artifacts \
#     -e AUTOTABML_LOG_FORMAT=json \
#     autotabml-studio:0.2.0
#
# Run (CLI):
#   docker run --rm -it autotabml-studio:0.2.0 autotabml --help
#   docker run --rm -it \
#     -v $(pwd)/data:/data:ro \
#     -v $(pwd)/artifacts:/app/artifacts \
#     autotabml-studio:0.2.0 \
#     autotabml benchmark /data/train.csv --target label

ARG PYTHON_VERSION=3.12

# ─── Stage 1: builder ──────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

WORKDIR /build

# Install uv into the builder image.
ENV UV_VERSION=0.11.23
RUN pip install --no-cache-dir "uv==${UV_VERSION}"

# Copy only what uv needs to resolve the lockfile (cache-friendly).
COPY pyproject.toml uv.lock .python-version ./

# Sync into a deterministic venv at /opt/venv.
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system
RUN uv sync --locked --no-install-project --group dev --all-extras

# Now copy the rest and install the project itself.
COPY . /build
RUN uv sync --locked --group dev --all-extras

# ─── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Create a non-root user (UID 10001) for the runtime.
RUN groupadd --system --gid 10001 autotabml \
    && useradd --system --uid 10001 --gid autotabml \
       --home /app --shell /bin/bash autotabml

# Copy the venv (and the project) from the builder.
COPY --from=builder --chown=autotabml:autotabml /opt/venv /opt/venv
COPY --from=builder --chown=autotabml:autotabml /build /app

WORKDIR /app

# Local-first defaults; can be overridden at run time.
ENV AUTOTABML_LOG_FORMAT=json \
    AUTOTABML_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv

# Artifact directory needs to be writable by the runtime user.
RUN mkdir -p /app/artifacts && chown -R autotabml:autotabml /app

USER autotabml

EXPOSE 8501

# Default command: start the Streamlit UI. Override to "autotabml ..."
# for headless CLI workflows.
CMD ["streamlit", "run", "app/main.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

# Container-level healthcheck.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=2).read()" \
    || exit 1
