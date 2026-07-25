#!/usr/bin/env bash
# Non-Docker production: 150-300 concurrent users (nginx + gunicorn)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -f .venv/bin/activate ]; then
  echo "[!] Run ./install.sh first" >&2
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate
# shellcheck source=/dev/null
source .env.nanoserve

CORES=$(nproc)
PROCS="${NANOSERVE_UVICORN_WORKERS:-4}"
export NANOSERVE_UVICORN_WORKERS="$PROCS"
export NANOSERVE_NUM_WORKERS="${NANOSERVE_NUM_WORKERS:-$((CORES / PROCS))}"
[ "$NANOSERVE_NUM_WORKERS" -lt 1 ] && export NANOSERVE_NUM_WORKERS=1
export NANOSERVE_MAX_BATCH="${NANOSERVE_MAX_BATCH:-32}"
export NANOSERVE_MAX_QUEUE="${NANOSERVE_MAX_QUEUE:-512}"
export PYTHONPATH="$ROOT"

echo "[*] Native 300-user mode: $PROCS gunicorn x $NANOSERVE_NUM_WORKERS engine workers"
exec "$ROOT/scripts/serve_production.sh"
