# NanoServe Audit Report — C++23, install.sh, Memory & GPU

**Date:** 2026-07-25  
**Tests:** **14/14 PASSED** (incl. 3 new memory stress tests)

---

## Verdict

| Layer | Status |
|-------|--------|
| C++23 engine | OK — `CMAKE_CXX_STANDARD 23`, `enum class`, `std::span`, `std::make_unique`, `std::string_view` |
| Buddy allocator (Rust) | OK — weights/scratch pools; alloc/free per request |
| CPU AVX2 | OK — deterministic, 200× infer stress pass |
| CUDA GEMV | OK — RTX 3060; 50× CPU/CUDA parity under load |
| OpenCL | Optional — not built on audit host |
| `install.sh` | OK — syntax valid; builds + venv + `.env.nanoserve` |
| Web UI / API | OK — HTTP tests pass |
| Memory leaks | No crash/leak under 200 CPU + 50 CUDA + 32 concurrent requests |

---

## Bugs Found & Fixed

| ID | Issue | Fix |
|----|-------|-----|
| M1 | `engine_create` leaked pools if weight alloc failed | Release pools on failure path |
| M2 | `engine_destroy_handle` unsafe on null pools/weights | Null guards before `pool_free`/`pool_release` |
| M3 | `engine_run_infer` no input validation | Guard null handle/backend/buffer |
| M4 | CUDA `cudaMalloc` unchecked | Check errors; rollback on partial alloc |
| M5 | `EngineWorker.probe_*` reloaded `.so` every call | `_LIB_CACHE` shared CDLL |
| M6 | `EnginePool` health probes re-loaded lib | Cache cuda/opencl in `_GpuState` |
| M7 | `install.sh` failed on re-run (venv exists) | Skip venv create if present |
| M8 | `install.sh` no post-build verify | Exit if `libnanoserve_engine.so` missing |
| P1 | Token string reallocation | `out.reserve(max_tokens * 16)` |

---

## Memory Architecture (verified)

```
pool_create (weights_pool 64KiB)  → int8 weights[1024]     [engine lifetime]
pool_create (scratch_pool 16MiB)  → float acts[1024]       [per request]
                                    pool_free after infer  [reuse, no fragmentation]
CUDA cudaMalloc                   → device buffers         [per backend handle, reused]
```

Scratch buffer returned to buddy pool after every `engine_infer` — 200 sequential calls stable.

---

## install.sh workflow

```bash
./install.sh                  # CPU build
ENABLE_CUDA=1 ./install.sh    # CUDA build
source .venv/bin/activate && source .env.nanoserve
python server/main.py
```

---

## Test commands

```bash
export LD_LIBRARY_PATH=$PWD/allocator/target/release
export NANOSERVE_ENGINE_LIB=$PWD/engine/build/libnanoserve_engine.so
export PYTHONPATH=$PWD
python3 tests/test_suite.py
```

---

## Remaining notes (non-blocking)

- OpenCL: install `ocl-icd-opencl-dev` + rebuild to enable
- CUDA 11.x + GCC 15: uses `engine/nvcc_wrapper.sh` + `g++-9`
- Python thread-local engine handles not auto-freed on thread exit — call `worker.cleanup()` for long-lived threads (pools use bounded thread pool in server)
