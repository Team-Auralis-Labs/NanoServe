# NanoServe — Extensive Test & Audit Report

**Date:** 2026-07-26  
**Environment:** Linux x86_64 · Pop!\_OS-class host  
**Repo:** `/home/anand/AURALIS_LABS/LLMOrchestrator/NanoServe`  
**Engine:** `libnanoserve_engine.so` (CUDA + CPU AVX2 enabled)  
**Emscripten:** 6.0.4 @ `/home/anand/App-Installers_and_configs/emsdk`

---

## Executive summary

| Category | Result |
|----------|--------|
| Unit & integration tests (48) | **PASS** |
| Valgrind (200 + 1000 cycles) | **PASS — 0 leaks** |
| Buddy allocator RSS audits | **PASS — plateau after warmup** |
| CPU / CUDA simulated parity | **PASS** |
| 200-user HTTP load (CPU/GPU/LAN) | **PASS — 100% success** |
| Server RSS under HTTP load | **PASS — +0.1 MiB delta** |
| WASM native buffer FFI | **PASS** |
| Emscripten browser build | **PASS — 125 KiB `.wasm`** |
| **Overall audit** | **PASS** |

NanoServe is **memory-safe**, **allocator-correct**, **network-servable**, and **WASM-buildable** on this host. Production scaling beyond single-process uvicorn should use `run_native_300.sh` (nginx + gunicorn).

---

## 1. Test matrix

### 1.1 Automated unit tests

| File | Tests | Result |
|------|-------|--------|
| `tests/test_suite.py` | 15 | PASS |
| `tests/test_gguf.py` | 8 | PASS |
| `tests/test_nanoq_loader.py` | 4 | PASS |
| `tests/test_simd_parity.py` | 3 | PASS |
| `tests/test_router_multimodel.py` | 3 | PASS |
| `tests/test_quantizer_fp16_fp4.py` | 3 | PASS |
| `tests/test_models_registry.py` | 3 | PASS |
| `tests/test_wasm_native.py` | 3 | PASS |
| `tests/test_wasm.py` | 5 | PASS (incl. Emscripten rebuild) |
| `tests/test_models_download.py` | 1 | PASS |
| **Total** | **48** | **ALL PASS** |

### 1.2 Integration highlights (`test_suite.py`)

- Quantizer int8 shape + `.nanoq` write
- CPU inference deterministic
- **CPU vs CUDA output parity** (identical token stream)
- GPU fallback when unavailable
- EnginePool CPU submit + GPU path
- SDK generate on `cpu` / `gpu` / `auto`
- Memory stress: 200 repeated inferences, 50-cycle CPU/CUDA parity under load, 32 concurrent pool jobs
- HTTP server: health, completions CPU/GPU, list models

---

## 2. Valgrind memory leak audit

**Command:** `./scripts/valgrind.sh`

### Harness A — 200 cycles (`tests/valgrind_engine.c`)

```
OK 200 cycles
HEAP SUMMARY: in use at exit: 0 bytes in 0 blocks
total heap usage: 8,409 allocs, 8,409 frees, ~16.8 GB moved
All heap blocks were freed -- no leaks are possible
ERROR SUMMARY: 0 errors from 0 contexts
```

### Harness B — 1000 cycles (`tests/valgrind_engine_ext.c`)

```
OK 1000 cycles
HEAP SUMMARY: in use at exit: 0 bytes in 0 blocks
total heap usage: 42,009 allocs, 42,009 frees, ~83.9 GB moved
All heap blocks were freed -- no leaks are possible
ERROR SUMMARY: 0 errors from 0 contexts
```

Each cycle exercises:

1. `engine_init()` → Rust `pool_create(64 MiB)` + `pool_create(16 MiB)`
2. `engine_infer()` → `pool_allocate(scratch)` → `pool_free(scratch)`
3. `engine_cleanup()` → `pool_release()` both arenas

**Verdict:** No definite, indirect, or reachable leaks in the C++/Rust FFI path.

Reports: `documentation/valgrind_report.txt`, `documentation/valgrind_report_extended.txt`

---

## 3. Rust buddy allocator ↔ C++23 safety

### Architecture

```
engine_core.cpp
  pool_create / pool_allocate / pool_free / pool_release
       ↓ extern "C"
allocator/src/lib.rs  (BuddyPool → Arena, power-of-2 blocks, buddy merge)
```

### Per-inference lifecycle

| Step | C++ call | Allocator action |
|------|----------|------------------|
| Engine create | `pool_create(64MB)`, `pool_create(16MB)` | Reserve buddy arenas |
| Synthetic weights | `pool_allocate(weights_pool, 1024)` | Demo int8 buffer |
| Each infer | `pool_allocate(scratch, 4KB)` | Activation buffer |
| End infer | `pool_free(scratch, …)` | Return block to buddy free list |
| Destroy | `pool_release()` × 2 | Drop arena when refcount zero |

`.nanoq` models store weights in `std::vector` inside `NanoqModel` — not buddy-pool heap for payload (only scratch per infer uses pool).

### RSS plateau tests

#### A. `tests/memory_rss_audit.py` (432 sequential inferences, 4 workers)

| Metric | Value |
|--------|-------|
| RSS after warmup (32 req) | 41.7 MiB |
| RSS after 400 more | 41.7 MiB |
| **Delta** | **+0.0 MiB** |
| Threshold | 48 MiB |
| **Result** | **PASS** |

#### B. `tests/memory_concurrent_audit.py` (8 workers, 15 bursts)

| Metric | Value |
|--------|-------|
| RSS after 1st burst | 42.0 MiB |
| RSS after 15 bursts | 42.4 MiB |
| Delta last 10 bursts | **+0.0 MiB** |
| **Result** | **PASS** |

#### C. `tests/memory_server_audit.py` (HTTP server, 170 requests)

| Round | Requests | OK | RSS |
|-------|----------|-----|-----|
| 1 (warmup) | 50 | 50/50 | 154.0 MiB |
| 2 | 40 | 40/40 | 154.1 MiB |
| 3 | 40 | 40/40 | 154.1 MiB |
| 4 | 40 | 40/40 | 154.1 MiB |

**Delta baseline → final:** +0.1 MiB · **PASS**

> Note: Absolute RSS (~154 MiB) is higher after prior 200-user load tests warmed engine threads; **growth after warmup** is what matters for leak detection.

---

## 4. CPU & GPU simulated tests

### Health probes (server `/health`)

- `gpu_cuda`: **true**
- `gpu_available`: **true**
- `native_available`: **true**
- `gguf_available`: **false** (optional extra not installed in this run)

### Device routing (single request)

| `device` | Response `device` | Notes |
|----------|-------------------|-------|
| `cpu` | `cpu` | ~26 ms latency |
| `gpu` | `cuda` | CUDA GEMV path active |
| `auto` | cuda or cpu | GPU when available |

### SIMD / parity

- `test_simd_parity.py`: int8, fp16, fp4 deterministic on CPU
- `test_cpu_cuda_output_parity`: bit-identical tokens CPU vs CUDA
- `test_cpu_cuda_parity_under_load`: 50 cycles, all match

**Verdict:** CPU and GPU backends produce consistent demo inference; no silent divergence.

---

## 5. 200-user load test

**Server:** single uvicorn on `0.0.0.0:8000`, `NANOSERVE_NUM_WORKERS=8`, micro-batcher (batch ≤32, 25 ms window)

**Tool:** `tests/load_test_report.py --users 200`

### Results

| Scenario | Endpoint | Device | Success | Throughput | p50 | p95 | max |
|----------|----------|--------|---------|------------|-----|-----|-----|
| Localhost | `127.0.0.1:8000` | cpu | **200/200** | 60.0 req/s | 1931 ms | 3210 ms | 3285 ms |
| Localhost | `127.0.0.1:8000` | gpu | **200/200** | 59.7 req/s | 1864 ms | 3237 ms | 3322 ms |
| **LAN network** | `192.168.20.15:8000` | cpu | **200/200** | 52.0 req/s | 2129 ms | 3734 ms | 3803 ms |

JSON reports: `/tmp/load_200_cpu.json`, `/tmp/load_200_gpu.json`, `/tmp/load_200_net.json`

### Network servability

- Server binds **`0.0.0.0:8000`** (all interfaces)
- 200 concurrent clients via LAN IP **192.168.20.15** succeeded with **100% success rate**
- Suitable for multi-client access on local network; for internet exposure add TLS + reverse proxy

### Latency interpretation

p50 ~1.9–2.1 s under 200 concurrent users is **expected** for single-process dev server:

- Queue depth + batch window (25 ms)
- 8 engine worker threads serializing GEMV
- Not representative of `run_native_300.sh` (nginx + 4× gunicorn)

---

## 6. WASM / Emscripten audit

### Build

```bash
source /home/anand/App-Installers_and_configs/emsdk/emsdk_env.sh
./scripts/build_wasm.sh
```

| Artifact | Size | Target | Result |
|----------|------|--------|--------|
| `nanoserve_engine.wasm` | **125 KiB** (128,215 B) | < 2 MB | **PASS** |
| `nanoserve_engine.js` | 13 KiB | — | OK |
| `assets/demo.nanoq` | 2.3 KiB | < 1 MB | OK |

Emscripten **6.0.4** · compile flags: `-O3 -std=c++23 -DNANOSERVE_WASM=1`, scalar SIMD fallbacks (no AVX in browser)

### WASM tests

| Test | Result |
|------|--------|
| `test_wasm_native.py` — buffer init/reload/parity | **PASS** |
| `test_wasm.py` — static bundle structure | **PASS** |
| `test_wasm.py` — `test_emscripten_build` (full rebuild) | **PASS** (7.1 s) |
| `test_wasm.py` — native buffer subprocess | **PASS** |

### Browser demo (manual verification steps)

```bash
npx serve deployment/wasm
# Open printed URL → WASM ready → load demo.nanoq → Generate
```

Exported FFI: `engine_init`, `engine_init_with_model_bytes`, `engine_reload_model_bytes`, `engine_infer`, `engine_model_info`, `engine_cleanup`

**WASM uses `buddy_pool_wasm.cpp`** (malloc-based stub) instead of Rust `libbuddy_alloc.so` — intentional for lean Emscripten linking; native path still uses Rust buddy allocator.

---

## 7. Findings & recommendations

### Strengths

1. **Zero Valgrind leaks** over 1,000 full init/infer/cleanup cycles
2. **Buddy allocator reuse** confirmed — RSS flat after warmup in all audits
3. **200/200 success** on CPU, GPU, and LAN for concurrent HTTP load
4. **CPU/CUDA parity** maintained under stress
5. **WASM build** completes; bundle **125 KiB** (well under budget)
6. Native Docker/GGUF/production paths **unchanged** by WASM tier

### Caveats

| Item | Severity | Note |
|------|----------|------|
| Single uvicorn for load test | Info | Use `run_native_300.sh` for production 200+ users |
| p95 latency ~3–3.7 s @ 200 users | Info | Queue + batching; tune workers/batch for SLA |
| GGUF not load-tested | Low | Optional; requires `[gguf]` extra + model |
| WASM buddy stub ≠ Rust allocator | Info | Browser-only; native still uses Rust |
| Emscripten PATH | Ops | Must `source emsdk_env.sh` before `./scripts/build_wasm.sh` |

### Recommended commands (regression)

```bash
# Full unit suite
for f in tests/test_*.py; do python3 "$f"; done

# Memory
./scripts/valgrind.sh
python3 tests/memory_rss_audit.py
python3 tests/memory_concurrent_audit.py

# Load (server running)
python3 tests/load_test_report.py --users 200 --device cpu
python3 tests/load_test_report.py --users 200 --device gpu

# WASM
source /home/anand/App-Installers_and_configs/emsdk/emsdk_env.sh
./scripts/build_wasm.sh
python3 tests/test_wasm.py
```

---

## 8. Audit verdict

**NanoServe PASSES this extensive audit** for:

- Memory safety (Valgrind)
- Rust buddy allocator integration (RSS plateau)
- CPU + CUDA simulation and parity
- 200-user concurrent HTTP (localhost + LAN)
- WASM Emscripten build and buffer FFI

No blocking defects found. System is **servable across the network** to multiple clients and **safe for continued development and deployment** on native/Docker paths, with WASM as an optional browser demo tier.

---

*Generated by automated audit run on 2026-07-26.*
