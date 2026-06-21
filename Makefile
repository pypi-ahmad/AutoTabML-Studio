# AutoTabML Studio developer Makefile
#
# All targets are deliberately simple; they exist to document the
# canonical commands and to make them discoverable on a fresh clone.

.PHONY: help install sync test test-cov lint format type-check security clean build doctor

PYTHON ?= 3.12.10

help:
	@echo "AutoTabML Studio - common dev commands"
	@echo "  make install      Create venv and sync all extras"
	@echo "  make sync         uv sync --locked with default + dev groups"
	@echo "  make test         Run unit tests (excludes integration)"
	@echo "  make test-cov     Run unit tests with coverage gate (>=65%)"
	@echo "  make lint         Run ruff check"
	@echo "  make format       Run ruff format (writes changes)"
	@echo "  make type-check   Run mypy on app/"
	@echo "  make security     Run bandit + pip-audit"
	@echo "  make build        Build wheel + sdist via hatchling"
	@echo "  make clean        Remove build artifacts and caches"
	@echo "  make doctor       Run autotabml doctor"

install:
	uv venv --python $(PYTHON)
	uv sync --locked --all-groups --all-extras

sync:
	uv sync --locked --all-groups

test:
	uv run --no-sync python -m pytest tests/ -q --tb=short

test-cov:
	uv run --no-sync python -m pytest tests/ --cov=app --cov-report=term --cov-fail-under=65 -q

lint:
	uv run --no-sync ruff check app/ tests/ scripts/

format:
	uv run --no-sync ruff format app/ tests/ scripts/

type-check:
	uv run --no-sync mypy app/ --config-file=pyproject.toml

security:
	uv run --no-sync bandit -r app/ -q || true
	uv run --no-sync pip-audit || true

build:
	uv run --no-sync python -m build

clean:
	rm -rf build/ dist/ .pytest_cache/ .ruff_cache/ .mypy_cache/ .coverage coverage.xml

doctor:
	uv run --no-sync autotabml doctor
