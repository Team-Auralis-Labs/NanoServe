# User Stress Report — NanoServe Load Testing

**Date:** 2026-07-25  
**Tool:** `tests/load_test_report.py`  
**Device:** CPU (AVX2)  
**Verdict:** <span class="badge">PASSED AT ALL TIERS</span>

---

## Executive summary

Concurrent HTTP load tests simulate real users hitting `/v1/completions` simultaneously. All three tiers — **50, 150, and 300 users** — completed with **100% success rate** and zero HTTP failures on the non-Docker native deployment.

---

## Test methodology

1. Start server with appropriate scaling profile (see deployment column).
2. Run `python3 tests/load_test_report.py --preset {50|150|300}`.
3. Each virtual user sends one completion request with `max_tokens=32`, `device=cpu`.
4. Metrics: success rate, throughput (req/s), latency percentiles.

---

## Results — 50 users (baseline)

| Metric | Value |
|--------|-------|
| Deployment | Single uvicorn (`run_native.sh`) |
| Success | 50 / 50 (100%) |
| Throughput | ~58–67 req/s |
| Latency p50 | ~400–550 ms |
| Latency p95 | ~700–850 ms |
| Wall clock | ~0.8 s |

Suitable for development and demos.

---

## Results — 150 users

| Metric | Value |
|--------|-------|
| Deployment | Single uvicorn, `NANOSERVE_NUM_WORKERS=$(nproc)` |
| Success | 150 / 150 (100%) |
| Throughput | ~60 req/s |
| Latency p50 | ~1800 ms |
| Latency p95 | ~2373 ms |
| Wall clock | ~2.5 s |

Acceptable for moderate load on a single process; consider production script for headroom.

---

## Results — 300 users (non-Docker)

| Metric | Value |
|--------|-------|
| Deployment | Single uvicorn (validated) / **`run_native_300.sh`** (recommended production) |
| Success | 300 / 300 (100%) |
| Throughput | **63.77 req/s** |
| Latency mean | 2792 ms |
| Latency p50 | 2506 ms |
| Latency p95 | **4508 ms** |
| Latency p99 | 4598 ms |
| Latency max | 4636 ms |
| Wall clock | 4.70 s |

Raw JSON: `documentation/load_300.json`

### Recommended 300-user production (non-Docker)

```bash
./install.sh
source .venv/bin/activate && source .env.nanoserve
./scripts/run_native_300.sh   # nginx :8000 + 4× gunicorn, workers = nproc/4 each
```

This matches Docker-free scaling documented in [SCALING.md](../SCALING.md) and splits engine workers across processes to avoid memory oversubscription.

---

## Comparison chart

| Tier | Users | Success % | req/s | p95 (ms) | Production script |
|------|-------|-----------|-------|----------|-------------------|
| Baseline | 50 | 100 | ~62 | ~800 | `run_native.sh` |
| Medium | 150 | 100 | ~60 | ~2373 | `run_native.sh` |
| High | 300 | 100 | ~64 | ~4508 | **`run_native_300.sh`** |

---

## How to reproduce

```bash
source .venv/bin/activate && source .env.nanoserve
./scripts/run_native.sh &          # or run_native_300.sh for 300-tier prod
sleep 2
python3 tests/load_test_report.py --preset 300
```

---

## Conclusion

NanoServe handles 300 simultaneous users on CPU with full success. Latency increases with concurrency as expected for a micro-batched inference server; production deployments should use **`./scripts/run_native_300.sh`** with nginx and tuned `NANOSERVE_MAX_QUEUE=512` for stable operation under sustained load.
