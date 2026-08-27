# TODO Phase 09 — TLS Advanced (Stretch)

> **Copy-paste this file into Agent mode to implement Phase 09.**
>
> **Status:** **Stretch** — implement after Phases 01–03 and preferably after Phase 06 (RAG co-host).
>
> **Master plan:** [TODO-NanoServe-Industry-grade-plan.md](../TODO-NanoServe-Industry-grade-plan.md) — Part C TLS-2, Part D device tiers
> **Prerequisite:** [TODO-Phase-03-TLS-Prefetch-Efficiency.md](TODO-Phase-03-TLS-Prefetch-Efficiency.md) Human checkpoint PASS
> **Index:** [TODO-Phase-INDEX.md](TODO-Phase-INDEX.md)

---

## Goal

Implement **TLS-2**: optional KV cache tiering/offload and **TLS-Train** (native streamed backward + gradient checkpointing for LoRA on edge) — enabling 7B-class Q4 inference on 4 GB SBCs and optional native retrain without PyTorch.

---

## Prerequisites

- Phase 03: TLS-1 prefetch + memory tests pass
- Phase 07 recommended if implementing TLS-Train
- Phase 06 optional for `test_tls_rag_budget.py`

---

## Scope

### KV cache offload (stretch)

When KV exceeds budget, tier oldest layers to mmap-backed storage (FlexGen-inspired). **Default if deferred:** cap `max_seq_len` instead.

```bash
NANOSERVE_TLS_KV_OFFLOAD=1   # only when implemented
```

### TLS-Train — streamed backward

| Technique | TLS integration |
|-----------|-----------------|
| Gradient checkpointing | Re-load chunk; recompute activations on backward |
| Optimizer state streaming | Adam moments per-chunk in `.nanoadapt` sidecar |
| LoRA only | Adapter resident; base weights streamed |
| Full fine-tune | GPU hosts opt-in only |

```text
for microbatch:
  forward chunks 0..N with checkpoint markers
  backward chunks N-1..0: reload → recompute → grad → update adapter
```

### Device tier targets (Part D)

| Tier | Target | TLS mode |
|------|--------|----------|
| T0 | Laptop 8–32 GB | chunk_layers=4–8 |
| T1 | SBC Pi 4/5 2–8 GB | chunk_layers=1–2; 7B Q4 qualitative |
| T2 | FPGA | Inference only — out of scope |
| T3 | MCU | Out of scope |

**7B Q4 on 4 GB Pi (T1) budget:** ~750–900 MB inference + optional RAG index

---

## Implementation steps

1. Design KV tier storage format; implement optional offload path
2. Implement TLS-Train backward loop in C++ engine (LoRA deltas only)
3. Stream optimizer state in `.nanoadapt` sidecar per chunk
4. Add `tests/test_tls_rag_budget.py`, `tests/test_tls_train_lora.py`
5. Manual benchmark doc for 7B Q4 on 4 GB profile (`chunk_layers=1`)
6. Document tier matrix in `documentation/` or phase sign-off

---

## Files to add/modify

**Modify:** `layer_stream.cpp`, `engine_core.cpp`, `nanoserve/train/spec.py`, adapter sidecar format

**New tests:** `tests/test_tls_rag_budget.py`, `tests/test_tls_train_lora.py`

---

## Automated verification

> **Post-build gate:** After **every** Phase 09 build, run **all four** subsections below before the human checkpoint. Do not start the next phase until every row in the verification matrix passes.

### 1. Unit & integration tests

```bash
export NANOSERVE_TLS=1
export NANOSERVE_TLS_CHUNK_LAYERS=1

python3 tests/test_tls_parity.py
python3 tests/test_tls_memory.py
python3 tests/test_tls_rag_budget.py
python3 tests/test_tls_train_lora.py
python3 tests/test_suite.py

bash scripts/audit_deployments.sh
```

### 2. Performance benchmarks

```bash
export NANOSERVE_TLS=1 NANOSERVE_TLS_CHUNK_LAYERS=1
# 7B Q4 qualitative run on 4 GB profile (manual or simulated cap)
/usr/bin/time -v python3 -c "
from nanoserve import Worker
w = Worker()
# w.load('models/7b-q4.nanoq', format='nanoq')
# w.infer('Stretch benchmark prompt', 64)
" 2>&1 | tee documentation/reports/PHASE09_BENCH.md || true
# Pass: coherent output; RSS within NANOSERVE_TLS_WEIGHT_ARENA_MB + KV
python3 tests/test_tls_rag_budget.py
python3 tests/test_tls_train_lora.py
```

### 3. Memory leak & RSS audits

```bash
export NANOSERVE_TLS=1 NANOSERVE_TLS_CHUNK_LAYERS=1
python3 tests/test_tls_memory.py
python3 tests/test_tls_rag_budget.py
./scripts/valgrind.sh
VALGRIND_CYCLES=2000 ./scripts/valgrind.sh || ./scripts/valgrind.sh
python3 tests/memory_rss_audit.py
# Pass: combined TLS+RAG RAM within env caps
```

### 4. Load & stress tests

```bash
export NANOSERVE_TLS=1 NANOSERVE_TLS_CHUNK_LAYERS=1
python3 tests/load_test_report.py --preset 50 --device cpu --out documentation/reports/PHASE09_LOAD.json
python3 tests/load_test_report.py --preset 150 --device cpu --max-tokens 8 --out documentation/reports/PHASE09_LOAD_150.json || true
# Pass: stretch items implemented → ≥98% at 50; document deferrals otherwise

bash scripts/audit_deployments.sh
```

### Post-build verification matrix

| Category | Command / artifact | Pass criteria |
|----------|-------------------|---------------|
| Unit / integration | `test_tls_rag_budget.py, test_tls_train_lora.py, test_tls_memory.py` | Stretch tests PASS or deferred |
| Performance | `7B Q4 manual benchmark doc` | RSS + coherence within caps |
| Memory leak / RSS | `extended valgrind + TLS+RAG budget tests` | Clean valgrind; combined RAM OK |
| Load / stress | `load_test_report.py --preset 50` | ≥98% or documented deferral |

**Sign-off:** Record results in `documentation/reports/PHASE09_VERIFY.md` (create if missing). CI must run sections 1–4 on every phase merge.

---

## Human checkpoint

| # | What you do | What you should see |
|---|-------------|---------------------|
| 1 | Run 7B Q4 with `chunk_layers=1` on 4 GB host (or simulated cap) | Qualitatively coherent output |
| 2 | RSS monitoring during 7B run | Within `NANOSERVE_TLS_WEIGHT_ARENA_MB` + KV budget |
| 3 | RAG + TLS co-hosted | Combined RAM within env caps (test or manual) |
| 4 | TLS-Train smoke (if implemented) | Adapter output ≠ base after streamed backward |
| 5 | KV offload (if implemented) | Longer ctx without OOM vs baseline |

**Stretch sign-off:** Mark PASS only for items actually implemented; document deferrals.

---

## Acceptance checklist

- [ ] **Post-build gate:** unit/integration + performance + memory leak/RSS + load/stress (see Automated verification); `PHASE09_VERIFY.md` recorded
- [ ] 7B-class Q4 qualitative on 4 GB host with `chunk_layers=1` (manual or CI profile)
- [ ] Peak RSS ≤ configured TLS + KV budget
- [ ] `tests/test_tls_rag_budget.py` passes (if RAG from Phase 06 available)
- [ ] `tests/test_tls_train_lora.py` passes (if TLS-Train implemented)
- [ ] KV offload OR documented deferral with `max_seq_len` cap
- [ ] No regression on TLS-0/1 parity and prefetch tests

---

## Do not break

- TLS-0/1 correctness
- QLoRA Python path from Phase 07 (TLS-Train is additive stretch)

---

## Next phase

[TODO-Phase-10-Platform-Stretch.md](TODO-Phase-10-Platform-Stretch.md)
---

## Appendix — TLS-2 + Part D + TLS file touch (full spec)

> Verbatim from [TODO-NanoServe-Industry-grade-plan.md](../TODO-NanoServe-Industry-grade-plan.md)

## Phase TLS-2 — KV tiering + TLS-Train (stretch)

### KV cache offload (stretch)

When KV exceeds budget, tier oldest layers to mmap-backed storage (FlexGen-inspired). Default v1: cap `max_seq_len` instead.

### TLS-Train — streamed backward on constrained devices

Align with Part B Phase T1 (QLoRA) but add **native C++ path** for edge hosts without PyTorch:

| Technique | TLS integration |
|-----------|-----------------|
| Gradient checkpointing | Re-load weight chunk during backward; recompute activations |
| Optimizer state streaming | Adam moments per-chunk in `.nanoadapt` sidecar; load/update/save per pass |
| LoRA only (v1 default) | Adapter weights resident; base weights streamed |
| Full fine-tune | Opt-in GPU hosts only |

```text
for microbatch:
  for chunk in 0..N:
    load chunk weights + adapter deltas
    forward with checkpoint markers
    unload chunk
  for chunk in N-1..0:
    load chunk
    recompute forward segment
    backward → grad accum for chunk
    update adapter (+ optional chunk optimizer state)
    save chunk state to disk
    unload chunk
```

**Acceptance:** `tests/test_tls_train_lora.py` — adapter changes output after streamed backward vs base.

---

## TLS file touch list (additive)

**New**

- `engine/include/layer_stream.hpp`, `engine/src/layer_stream.cpp`
- `engine/include/activation_buffer.hpp`, `engine/src/activation_buffer.cpp`
- `engine/src/prefetch.cpp`
- `tests/test_tls_parity.py`, `tests/test_tls_memory.py`, `tests/test_tls_prefetch.cpp`
- `tests/test_tls_rag_budget.py`, `tests/test_tls_train_lora.py`

**Modify**

- `engine/include/nanoq_archive.hpp` — chunk index fields
- `engine/src/engine_core.cpp` — TLS forward path
- `nanoserve/quantizer/export_hf.py` — chunk-aware export
- `rust/nanoq_runtime/src/validate.rs` — chunk bounds + optional chunk_hash


# PART D — Device tier matrix

| Tier | Target | RAM | TLS mode | RAG co-host | Training |
|------|--------|-----|----------|-------------|----------|
| **T0** | Laptop / x86_64 Linux | 8–32 GB | mmap + prefetch; `chunk_layers=4–8` | Full HNSW + 7B TLS | QLoRA via `[train]` extra |
| **T1** | SBC (Pi 4/5) | 2–8 GB | `chunk_layers=1–2`; int8/fp4; NVMe | Index + TLS 7B Q4 | LoRA rank ≤4 CPU |
| **T2** | FPGA (Zynq, Agilex) | BRAM + DDR | DMA weight chunk → BRAM; fixed pipeline | Inference only v1 | Out of scope v1 |
| **T3** | MCU (H7 + PSRAM) | 1–2 MB SRAM + PSRAM | QSPI flash stream; int8; SmolLM-135M class | Out of scope v1 | Out of scope v1 |

### RAM budget formula (inference + optional RAG)

```text
peak_ram ≈ tls_weight_arena
         + kv_cache(n_layers, seq_len, hidden, n_kv_heads)
         + activation_buffer(batch, seq, hidden)
         + resident_embed_head
         + rag_index (optional, mmap — mostly page cache)
         + rag_chunk_cache (NANOSERVE_CORPUS_CHUNK_CACHE_MB)
```

### Example: 7B Q4 on 4 GB Pi (T1)

| Component | Estimate |
|-----------|----------|
| TLS weight chunk (1 layer, Q4) | ~120–150 MB |
| KV cache (32 layers, 2048 ctx, fp16) | ~512 MB |
| Activations + embed/head | ~50 MB |
| RAG index (10k chunks, int8 384-d) | ~6 MB mmap |
| Hot chunk cache | 64 MB configurable |
| **Total** | ~750 MB–900 MB inference + RAG — feasible on 4 GB with OS headroom |

# Unified testing strategy (additive)

| Test | File | Validates |
|------|------|-----------|
| TLS parity | `tests/test_tls_parity.py` | Chunked forward == full-model forward (logits Δ < ε) |
| TLS memory cap | `tests/test_tls_memory.py` | RSS ≤ weight_chunk + KV + margin |
| TLS prefetch | `tests/test_tls_prefetch.cpp` | No correctness regression; latency improvement |
| TLS + RAG cohost | `tests/test_tls_rag_budget.py` | Combined RAM within env caps |
| TLS-Train smoke | `tests/test_tls_train_lora.py` | Adapter changes output after streamed backward |

See also Part A [Testing strategy](#testing-strategy) and Part B [Testing strategy](#testing-strategy-1).

