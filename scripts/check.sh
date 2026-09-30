#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

uv sync --locked --dev
uv run --no-sync --offline ruff check .
uv run --no-sync --offline mypy .
uv run --no-sync --offline pytest
