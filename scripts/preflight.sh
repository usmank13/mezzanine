#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python"
  fi
fi

echo "[preflight] ruff format --check"
"$PYTHON" -m ruff format --check .

echo "[preflight] ruff check"
"$PYTHON" -m ruff check .

echo "[preflight] pytest"
"$PYTHON" -m pytest -q
