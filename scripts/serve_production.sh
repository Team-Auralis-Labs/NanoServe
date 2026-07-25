#!/usr/bin/env bash
# Production serve: gunicorn + uvicorn workers + optional nginx (150–300 users)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source .venv/bin/activate 2>/dev/null || true
# shellcheck source=/dev/null
[ -f .env.nanoserve ] && source .env.nanoserve

PROCESSES="${NANOSERVE_UVICORN_WORKERS:-4}"
BASE_PORT="${BASE_PORT:-8001}"
CORES=$(nproc)
PER_WORKER="${NANOSERVE_NUM_WORKERS:-$((CORES / PROCESSES))}"
[ "$PER_WORKER" -lt 1 ] && PER_WORKER=1
export NANOSERVE_NUM_WORKERS="$PER_WORKER"
export NANOSERVE_MAX_BATCH="${NANOSERVE_MAX_BATCH:-32}"
export NANOSERVE_MAX_QUEUE="${NANOSERVE_MAX_QUEUE:-512}"
export PYTHONPATH="$ROOT"

echo "[*] NanoServe production (non-Docker)"
echo "    processes=$PROCESSES x engine_workers=$NANOSERVE_NUM_WORKERS (= $((PROCESSES * NANOSERVE_NUM_WORKERS)) total)"
echo "    max_batch=$NANOSERVE_MAX_BATCH  max_queue=$NANOSERVE_MAX_QUEUE"

pip install -q gunicorn uvicorn 2>/dev/null || true

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 0 $((PROCESSES - 1))); do
  PORT=$((BASE_PORT + i))
  echo "[*] Starting worker on :$PORT"
  PORT=$PORT gunicorn server.main:app \
    -k uvicorn.workers.UvicornWorker \
    -w 1 \
    -b "127.0.0.1:$PORT" \
    --timeout 120 \
    --graceful-timeout 30 &
  PIDS+=($!)
done

sleep 2

if command -v nginx >/dev/null 2>&1 && [ "${USE_NGINX:-1}" = "1" ]; then
  echo "[*] nginx front door :8000 -> ports $BASE_PORT–$((BASE_PORT + PROCESSES - 1))"
  nginx -c "$ROOT/deployment/nginx.conf" -g "daemon off;" &
  PIDS+=($!)
else
  echo "[*] Single-process mode (no nginx). Use: uvicorn server.main:app --host 0.0.0.0 --port 8000"
  wait
fi

wait
