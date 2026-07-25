# NanoServe Scaling Guide (150–300+ users)

Non-Docker scaling uses the same FastAPI micro-batcher; throughput comes from **engine worker threads** and, at 300 users, **multiple gunicorn processes behind nginx**.

## Tiers

| Tier | Concurrent users | Script | nginx |
|------|------------------|--------|-------|
| Dev | ≤50 | `./scripts/run_native.sh` | No |
| Production | 150 | `./scripts/run_native.sh` | Optional |
| **High load** | **300** | **`./scripts/run_native_300.sh`** | **Yes** |

Prior requirements for non-Docker production: [REQUIREMENTS.md](REQUIREMENTS.md).

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `NANOSERVE_NUM_WORKERS` | `nproc` (dev) or `nproc/4` (300 prod) | Engine threads **per process** |
| `NANOSERVE_MAX_BATCH` | 32 | Micro-batch size |
| `NANOSERVE_BATCH_WINDOW_S` | 0.025 | Batch collect window |
| `NANOSERVE_MAX_QUEUE` | 512 (256 in dev) | HTTP 503 when exceeded |
| `NANOSERVE_UVICORN_WORKERS` | 4 | Gunicorn processes (`run_native_300.sh`) |

`install.sh` writes defaults to `.env.nanoserve`. **`run_native_300.sh` overrides worker counts** for correct 300-user production.

## Stage 1 — 150 users (single node)

```bash
source .venv/bin/activate && source .env.nanoserve
./scripts/run_native.sh

python3 tests/load_test_report.py --preset 150 --device cpu
```

## Stage 2 — 300 users (non-Docker production)

```bash
source .venv/bin/activate && source .env.nanoserve
chmod +x scripts/run_native_300.sh scripts/serve_production.sh
./scripts/run_native_300.sh
# nginx :8000 → gunicorn :8001-8004, each with nproc/4 engine workers

python3 tests/load_test_report.py --preset 300 --device cpu --url http://127.0.0.1:8000
```

Requires **nginx** (`sudo apt-get install nginx`) and **gunicorn** (installed by production script).

Manual equivalent:

```bash
export NANOSERVE_UVICORN_WORKERS=4
export NANOSERVE_NUM_WORKERS=$(($(nproc) / 4))
export NANOSERVE_MAX_QUEUE=512
./scripts/serve_production.sh
```

## Verified results (CPU, native)

| Preset | Success | Throughput | p95 latency |
|--------|---------|------------|-------------|
| 50 | 100% | ~62 req/s | ~800 ms |
| 150 | 100% | ~60 req/s | ~2373 ms |
| 300 | 100% | ~64 req/s | ~4508 ms |

Details: [reports/STRESS_REPORT.md](reports/STRESS_REPORT.md).

## Memory validation (Valgrind)

```bash
./scripts/valgrind.sh
```

Report: [reports/VALGRIND_REPORT.md](reports/VALGRIND_REPORT.md).

## Docker

Scale replicas in `docker-compose.yml` with nginx fronting multiple `nanoserve` containers (same pattern as native 300-user tier).
