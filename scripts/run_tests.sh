#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Project virtualenv not found at $ROOT_DIR/.venv"
  echo "Create it first with: make install"
  exit 1
fi

exec "$VENV_PYTHON" -m pytest -q -m "not live_llm" "$@"
