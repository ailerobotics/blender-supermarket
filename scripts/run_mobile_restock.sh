#!/usr/bin/env bash
# Explicit runtime remains guarded inside Python. CPU checks do not probe the GPU.
# GNU timeout owns only this launch's process group, never unrelated applications.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESTOCK_PYTHON="${RESTOCK_PYTHON:-python3}"
exec timeout --signal=TERM --kill-after=15s 1200s \
  "$RESTOCK_PYTHON" "$SCRIPT_DIR/run_mobile_restock.py" "$@"
