# TODO Phase 03 — TLS Prefetch + Efficiency

> **Copy-paste this file into Agent mode to implement Phase 03.**
>
> **Master plan:** [TODO-NanoServe-Industry-grade-plan.md](../TODO-NanoServe-Industry-grade-plan.md) — Part C TLS-1, Part A Phase 5
> **Prerequisite:** [TODO-Phase-02-Export-Orchestrator-TLS0.md](TODO-Phase-02-Export-Orchestrator-TLS0.md) Human checkpoint PASS
> **Index:** [TODO-Phase-INDEX.md](TODO-Phase-INDEX.md)

---

## Goal

Complete **TLS-1** (double-buffer prefetch, memory budget enforcement) and **Phase 5 efficiency** (mmap weights, buddy KV, batching tuning) so layer streaming meets RSS targets without correctness regression.

---

## Prerequisites

- Phase 02: TLS-0 parity passes
- Export + native inference working

---

## Scope

### Phase TLS-1 — Prefetch + memory budget

**Double-buffer pattern:**

```text
Buffer A: computing chunk k
Buffer B: async prefetch chunk k+1 from mmap
On chunk k done: swap A↔B, signal prefetch k+2
```

**Environment variables:**

```bash
NANOSERVE_TLS=1
NANOSERVE_TLS_CHUNK_LAYERS=2
NANOSERVE_TLS_PREFETCH=1
NANOSERVE_TLS_WEIGHT_ARENA_MB=512
NANOSERVE_TLS_KV_OFFLOAD=0    # stretch — defer to Phase 09
```

Implement `engine/src/prefetch.cpp` with background `std::thread`.

### Phase 5 — Efficiency & low-resource profile

| Technique | Implementation |
|-----------|----------------|
| mmap weights | `mmap()` archive; no full-RAM copy |
| Buddy pool KV | Reuse `allocator/` for K/V growth |
| Layer-wise peak RAM | TLS chunk rotation (from Phase 02) |
| Batching | Micro-batcher prefill when prompts align |
| Threading | `NANOSERVE_NUM_WORKERS` — one graph per worker |
| Defaults | int8 v3, `max_seq_len=2048`, LRU `NANOSERVE_MAX_LOADED_MODELS=2` |

**Target budgets (distilgpt2-class, int8):**

| Metric | Target |
|--------|--------|
| Weights on disk | ~80–120 MB |
| RAM at inference | mmap + KV ~50–150 MB @ 2048 ctx |
| Latency vs GGUF Q4 | within 1.5× on same CPU |

---

## Implementation steps

1. Implement ping-pong weight buffers in `LayerStreamScheduler`
2. Add prefetch worker thread; sync before chunk boundary
3. Enforce `NANOSERVE_TLS_WEIGHT_ARENA_MB` via buddy pool limits
4. Audit mmap paths — no full tensor copy into RAM
5. Tune micro-batcher for same-model prefill batching
6. Add `tests/test_tls_memory.py`, `tests/test_tls_prefetch.cpp`, `tests/test_nanoq_memory.py`
7. Verify prefetch on/off produces identical token sequences

---

## Files to add/modify

**New:** `engine/src/prefetch.cpp`, `tests/test_tls_memory.py`, `tests/test_tls_prefetch.cpp`, `tests/test_nanoq_memory.py`

**Modify:** `layer_stream.cpp`, `engine_core.cpp`, `k_cache.cpp`, micro-batcher in `server/main.py` (if needed)

---

## Automated verification

> **Post-build gate:** After **every** Phase 03 build, run **all four** subsections below before the human checkpoint. Do not start the next phase until every row in the verification matrix passes.

### 1. Unit & integration tests

```bash
export NANOSERVE_TLS=1
export NANOSERVE_TLS_PREFETCH=1
export NANOSERVE_TLS_WEIGHT_ARENA_MB=512

python3 tests/test_tls_parity.py
python3 tests/test_tls_memory.py
# ./engine/build/test_tls_prefetch
python3 tests/test_nanoq_memory.py
python3 tests/test_suite.py

bash scripts/audit_deployments.sh
curl -s localhost:8000/health | jq .
```

### 2. Performance benchmarks

```bash
export NANOSERVE_TLS=1 NANOSERVE_TLS_PREFETCH=0
/usr/bin/time -f 'prefetch_off wall=%e maxrss=%M' python3 -c "
from nanoserve import Worker
w = Worker(); w.load('models/distilgpt2-int8.nanoq', format='nanoq')
w.infer('benchmark prefetch off', 64)
"

export NANOSERVE_TLS_PREFETCH=1
/usr/bin/time -f 'prefetch_on wall=%e maxrss=%M' python3 -c "
from nanoserve import Worker
w = Worker(); w.load('models/distilgpt2-int8.nanoq', format='nanoq')
w.infer('benchmark prefetch on', 64)
"
# Pass: prefetch_on wall ≤ prefetch_off (or documented platform exception)
# Pass: RSS ≤ NANOSERVE_TLS_WEIGHT_ARENA_MB + KV + margin (test_tls_memory.py)
```

### 3. Memory leak & RSS audits

```bash
export NANOSERVE_TLS=1 NANOSERVE_TLS_PREFETCH=1
python3 tests/test_tls_memory.py
python3 tests/test_nanoq_memory.py
./scripts/valgrind.sh
python3 tests/memory_rss_audit.py
python3 tests/memory_concurrent_audit.py
# Pass: no RSS growth after warmup under TLS path
```

### 4. Load & stress tests

```bash
export NANOSERVE_TLS=1 NANOSERVE_TLS_PREFETCH=1
python3 tests/load_test_report.py --preset 50 --device cpu --out documentation/reports/PHASE03_LOAD.json
python3 tests/load_test_report.py --preset 150 --device cpu --out documentation/reports/PHASE03_LOAD_150.json || true
# Pass: ≥98% success at 50 users; p95 documented; no OOM under TLS caps

bash scripts/audit_deployments.sh
```

### Post-build verification matrix

| Category | Command / artifact | Pass criteria |
|----------|-------------------|---------------|
| Unit / integration | `test_tls_parity.py, test_tls_memory.py, test_nanoq_memory.py` | All PASS |
| Performance | `prefetch on/off benchmark` | Prefetch ≤ latency or documented |
| Memory leak / RSS | `valgrind.sh + TLS memory tests` | RSS within arena + KV budget |
| Load / stress | `load_test_report.py --preset 50 (150 optional)` | ≥98% success |

**Sign-off:** Record results in `documentation/reports/PHASE03_VERIFY.md` (create if missing). CI must run sections 1–4 on every phase merge.

---

## Human checkpoint

| # | What you do | What you should see |
|---|-------------|---------------------|
| 1 | Run inference with TLS + prefetch ON | Same text output as TLS without prefetch |
| 2 | Run `test_tls_memory.py` | RSS ≤ weight arena + KV + margin |
| 3 | `curl /health` | Server healthy; model count unchanged |
| 4 | Monitor `/usr/bin/time -v` or test RSS cap | Peak RAM within Phase 5 budget for distilgpt2 |
| 5 | `audit_deployments.sh` | Still passes |

---

## Acceptance checklist

- [ ] **Post-build gate:** unit/integration + performance + memory leak/RSS + load/stress (see Automated verification); `PHASE03_VERIFY.md` recorded
- [ ] `tests/test_tls_memory.py` passes
- [ ] Prefetch does not change token outputs vs non-prefetch
- [ ] mmap used for weights (no full-RAM copy verified in tests or profiling)
- [ ] Memory footprint ≤ GGUF Q4 for same model class (qualitative)
- [ ] `/health` and existing completions unchanged
- [ ] `audit_deployments.sh` passes

---

## Do not break

- TLS-0 parity; GGUF path; v2 demo; API contract

---

## Next phase

[TODO-Phase-09-TLS-Advanced-Stretch.md](TODO-Phase-09-TLS-Advanced-Stretch.md) (after RAG/train tracks or in parallel)

Recommended next for RAG track: [TODO-Phase-04-NMDP-Sandbox.md](TODO-Phase-04-NMDP-Sandbox.md)
---

## Appendix — Phase 5 + TLS-1 (full spec)

> Verbatim from [TODO-NanoServe-Industry-grade-plan.md](../TODO-NanoServe-Industry-grade-plan.md)

## Phase 5 — Efficiency & low-resource profile

| Technique | Implementation |
|-----------|----------------|
| **mmap weights** | `mmap()` archive; no full-RAM copy |
| **Buddy pool KV** | Reuse `allocator/` Rust buddy for K/V growth |
| **Layer-wise peak RAM** | Optional: load attention weights on demand for fp16 fallback |
| **Batching** | Extend micro-batcher to batch prefill when prompts align (same model) |
| **Threading** | `NANOSERVE_NUM_WORKERS` — one graph per worker; no GIL |
| **Defaults** | int8 v3, `max_seq_len=2048`, LRU `NANOSERVE_MAX_LOADED_MODELS=2` |

**Target budgets (distilgpt2-class, int8):**

| Metric | Target |
|--------|--------|
| Weights on disk | ~80–120 MB |
| RAM at inference | mmap + KV ~50–150 MB @ 2048 ctx |
| Latency vs GGUF Q4 | within 1.5× on same CPU (v1 goal) |

---


> **TLS cross-link (Phase 5):** Phase 5 "Layer-wise peak RAM" is the entry point for **Temporal Layer Streaming**. See [Part C — TLS implementation deep-dive](#part-c-tls-implementation-deep-dive) and [Phase TLS-0 / TLS-1](#phase-tls-0--chunk-index--correctness) in the unified build order. Env: `NANOSERVE_TLS_CHUNK_LAYERS`, `NANOSERVE_TLS_PREFETCH`, `NANOSERVE_TLS_WEIGHT_ARENA_MB`.

## Phase TLS-1 — Prefetch + memory budget

### Double-buffer pattern

```text
Buffer A: computing chunk k
Buffer B: async prefetch chunk k+1 from mmap
On chunk k done: swap A↔B, signal prefetch k+2
```

### Environment variables

```bash
NANOSERVE_TLS=1                           # enable layer streaming (default 1 when RAM budget exceeded)
NANOSERVE_TLS_CHUNK_LAYERS=2              # layers per chunk (1 for tightest RAM)
NANOSERVE_TLS_PREFETCH=1                  # double-buffer next chunk
NANOSERVE_TLS_WEIGHT_ARENA_MB=512         # max buddy slab for weight chunk
NANOSERVE_TLS_KV_OFFLOAD=0                # stretch: page KV tiers to flash
```

**Acceptance:** `tests/test_tls_memory.py` — RSS ≤ `NANOSERVE_TLS_WEIGHT_ARENA_MB` + KV + activation margin; `tests/test_tls_prefetch.cpp` — no correctness regression vs non-prefetch.

---
