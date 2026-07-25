#!/usr/bin/env bash
# Non-Docker dev/single-node: up to ~150 users (single uvicorn)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source .venv/bin/activate
# shellcheck source=/dev/null
source .env.nanoserve
export NANOSERVE_NUM_WORKERS="${NANOSERVE_NUM_WORKERS:-$(nproc)}"
export NANOSERVE_MAX_BATCH="${NANOSERVE_MAX_BATCH:-32}"
export NANOSERVE_MAX_QUEUE="${NANOSERVE_MAX_QUEUE:-256}"
export PYTHONPATH="$ROOT"
exec python3 -m uvicorn server.main:app --host 0.0.0.0 --port "${PORT:-8000}"
