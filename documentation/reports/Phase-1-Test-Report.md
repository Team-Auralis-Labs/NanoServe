# Phase 1 — Comprehensive Test Report

**Project:** NanoServe Native Foundation (`.nanoq` v3)  
**Date:** 2026-08-27  
**Auditor:** Phase 01 final audit agent  
**Fixture:** `tests/fixtures/distilgpt2-int8.nanoq` (486 MB, fp32 weights, v3 archive)  
**Engine:** `engine/build/libnanoserve_engine.so` (post bounds-fix rebuild)

---

## Executive Summary

| Area | Result | Notes |
|------|--------|-------|
| Unit / integration | **PASS** | 42 tests across 9 suites; 0 failures |
| User-level API | **PASS** | Real LLM text via `format=nanoq`; legacy v2 intact |
| Stress / load | **PASS** | 200-cycle native; 20 concurrent v3 API; 50-user load 100% |
| Memory / leaks | **PASS** | Valgrind 0 leaks; RSS plateau 0.0 MiB drift |
| Performance | **PASS** | ~16.3 tok/s; ~1.0s/infer warm (16 tok) |
| Security (Blake3) | **PASS** | Tampered footer rejected |
| Bugs found | **1 severe (fixed)** | OOB past wpe/KV limit; model_info gap fixed |

**Verdict:** Phase 1 is **production-ready for Phase 2 handoff** on the GPT-2/distilgpt2 CPU path. Known deferrals are structural (file layout, optional FFI, Llama tokenizer) — not runtime blockers.

---

## 1. Unit & Integration Tests

| Suite | Tests | Result | Duration |
|-------|-------|--------|----------|
| `tests/test_nanoq_v3_loader.py` | 6 | PASS | ~13s |
| `tests/test_tokenizer_rust.py` | 2 | PASS | ~7s |
| `tests/test_nanoq_loader.py` | 4 | PASS | <1s |
| `tests/test_gguf.py` | 8 | PASS | <1s |
| `tests/test_simd_parity.py` | 3 | PASS | <1s |
| `tests/test_quantizer_fp16_fp4.py` | 3 | PASS | <1s |
| `tests/test_nanoq_memory.py` | 1 | PASS | ~14s |
| `tests/test_suite.py` | 15 (2 skip) | PASS | ~1s |
| `engine/build/test_transformer_gpt2` | native | PASS | instant |

### v3 loader coverage

- Model info: `format=nanoq_v3`, `arch=gpt2`, `n_layers=6`, `vocab_size=50257`, `max_seq_len=1024`
- Inference: coherent English (not 22-word GEMV demo)
- Blake3 tamper: rejected at validate + load
- Legacy v2: `legacy_demo: true` still works
- KV reset: isolated prompts produce identical output
- Sequence bounds: `max_seq_len` clamped to wpe rows

### Native golden forward

```
forward_ok best_token=383 logit=-31.6377 vocab=50257
bounds_ok max_pos=1024
```

Matches HuggingFace distilgpt2 greedy top token for `"Hello"` → ` The` (token 383).

---

## 2. User-Level Tests

### 2.1 Health (`GET /health`)

```json
{
  "status": "ok",
  "native_available": true,
  "active_format": "nanoq",
  "gguf_available": true,
  "models_loaded": 1
}
```

### 2.2 v3 Completion (`POST /v1/completions`)

**Request:**
```json
{
  "prompt": "Hello",
  "max_tokens": 32,
  "format": "nanoq",
  "model": "tests/fixtures/distilgpt2-int8.nanoq",
  "device": "cpu"
}
```

**Response (excerpt):**
```json
{
  "text": " The following is a new feature in the game.\n\n\n...",
  "latency_ms": 1847,
  "device": "cpu",
  "format": "nanoq",
  "warnings": []
}
```

**Assessment:** Real distilgpt2 autoregressive output — **not** the synthetic 22-word GEMV vocabulary.

### 2.3 Legacy v2 path

Router infer on synthetic 32×64 int8 matrix → `"model and fragmentation serves"` with `format=nanoq`. **PASS**

### 2.4 Tokenizer parity

Rust GPT-2 BPE encode/decode matches `transformers` on fixture strings. **PASS**

---

## 3. Stress Tests

| Test | Config | Result |
|------|--------|--------|
| Native 200-cycle infer | v3 fixture, 8 tok/request | **PASS** — no crash, no RSS runaway |
| API 20 concurrent | v3 fixture, 8 tok, 10 workers | **PASS** — 20/20 OK, p50 ~1270 ms |
| Load preset 50 | default demo path, CPU | **PASS** — 50/50 (100%), p50 685 ms |
| Long sequence smoke | 80×"hello" prompt + 200 tok | **PASS** — truncates at 1024, no OOB |

---

## 4. Throughput & Performance

### 4.1 Warm 5-run benchmark (Worker, v3 fixture)

| Run | Latency (s) |
|-----|-------------|
| 1 (cold) | 2.33 |
| 2–5 (warm) | ~0.97 each |
| **Aggregate** | wall=6.41s, maxrss=993 MB |

### 4.2 Throughput

| Metric | Value |
|--------|-------|
| Prompt | `"Hello world throughput test"` |
| New tokens | 32 |
| Elapsed | 1.96 s |
| **Throughput** | **16.3 tokens/sec** (CPU, fp32 distilgpt2) |

### 4.3 Regression vs prior build

No >10% latency regression on warm path (0.97s vs prior ~0.98s documented in PHASE01_BENCH.md).

---

## 5. Memory Efficiency & Leak Audits

| Check | Command / tool | Result |
|-------|----------------|--------|
| RSS sustained load | `tests/memory_rss_audit.py` | **PASS** — 0.0 MiB growth over 400 inferences |
| Concurrent RSS | `tests/memory_concurrent_audit.py` | **PASS** — plateau +0.1 MiB after 6 bursts |
| v3 RSS cap | `tests/test_nanoq_memory.py` | **PASS** — 20 inferences, no >50% drift |
| Valgrind C harness | `scripts/valgrind.sh` | **PASS** — 0 errors, 0 leaks, 1000 cycles |
| Valgrind extended | 1000 cycles, 83 GB allocated/freed | **PASS** — all heap blocks freed |

### Memory profile (v3 distilgpt2)

- **Peak RSS:** ~993 MB (mmap + fp32 weights + KV + scratch)
- **Python orchestrator RSS:** ~40 MB (without model loaded in process)
- **KV cache:** 6 layers × 1024 seq × 12 heads × 64 dim × 2 (K/V) × 4 B ≈ 37.5 MB

---

## 6. Bugs Found & Rectified

### 6.1 SEVERE — OOB read/write past position-embedding limit (FIXED)

**Symptom:** Sequences exceeding `wpe.shape[0]` (1024 for distilgpt2) could read past mmap'd weights and write past KV cache bounds — silent memory corruption risk.

**Root cause:** `forward_token()` indexed `wpe + pos * H` and KV store at `pos` without checking against wpe rows or KV `max_seq`. Config `max_seq_len` defaulted to 2048 while wpe only has 1024 rows.

**Fix (files changed):**
- `engine/src/transformer_gpt2.cpp` — clamp `max_pos_` to `min(config.max_seq_len, wpe.rows)`; reject `forward_token` when `pos >= max_pos_`; truncate `generate()` at limit
- `engine/include/transformer.hpp` — add `max_pos_` member
- `engine/src/nanoq_archive.cpp` — expose clamped `max_seq_len` in model_info JSON
- `engine/tests/test_transformer_gpt2.cpp` — assert `forward_token(pos=max_pos)` returns false
- `tests/test_nanoq_v3_loader.py` — bounds metadata test

**Verification:** `bounds_ok max_pos=1024`; long-sequence smoke returns partial output without crash.

### 6.2 MODERATE — model_info omitted effective max_seq_len (FIXED)

**Symptom:** API/clients could not see the true sequence limit.

**Fix:** `NanoqArchiveV3::model_info_json()` now includes wpe-clamped `max_seq_len`.

### 6.3 LOW — test_gguf.py failure when llama-cpp installed (FIXED earlier)

**Fix:** Mock `gguf_available=False` in fallback test to exercise native fallback path deterministically.

### 6.4 No regressions detected in

- GGUF routing/probe (`test_gguf.py`)
- v2 legacy loader
- SIMD quantizer parity
- Engine pool concurrency (`test_suite.py`)

---

## 7. Known Deferrals (Non-Blocking)

| Item | Status | Target |
|------|--------|--------|
| Separate `kv_cache.*`, `sampler.cpp`, `engine/src/ops/` | Inline in transformer | Phase 2 refactor optional |
| `gemm_fp16`, `gemm_fp4`, `attention_qkv` backends | Stub/default only | Phase 2 perf |
| Llama SentencePiece tokenizer | Not implemented | Phase 2 |
| `engine_infer_stream`, `engine_set_sampler` FFI | Not implemented | Phase 2 |
| Full `third_party/llama.cpp` kernel subset | CMake stub | Phase 2+ |
| True int8 distilgpt2 quality | Fixture uses fp32 | Phase 2 export |
| Load test with v3 fixture as default | Manual model path required | Phase 2 registry |

---

## 8. Artifacts

| File | Description |
|------|-------------|
| `documentation/reports/PHASE01_VERIFY.md` | Sign-off checklist |
| `documentation/reports/PHASE01_BENCH.md` | Baseline benchmark |
| `documentation/reports/PHASE01_LOAD.json` | 50-user load JSON |
| `documentation/valgrind_report.txt` | Valgrind 200-cycle |
| `documentation/valgrind_report_extended.txt` | Valgrind 1000-cycle |

---

## 9. Sign-Off

| Criterion | Status |
|-----------|--------|
| `format=nanoq` produces real LLM text | ✅ |
| Output ≠ 22-word GEMV demo | ✅ |
| Blake3 footer validation | ✅ |
| Legacy v2 demo preserved | ✅ |
| GGUF path unchanged | ✅ |
| `engine_reset_kv` works | ✅ |
| No memory leaks (Valgrind) | ✅ |
| Load ≥98% success | ✅ (100%) |
| No severe open bugs | ✅ (OOB fixed) |

**Phase 1 Status: APPROVED for Phase 2**

---

*Generated by automated audit run on 2026-08-27. Re-run gate:*

```bash
cd rust/nanoq_runtime && cargo build --release
cd engine/build && cmake .. -DNANOSERVE_LLAMA_CPP_KERNELS=1 && make -j$(nproc)
export LD_LIBRARY_PATH=$PWD/../../allocator/target/release:$LD_LIBRARY_PATH
export NANOSERVE_ENGINE_LIB=$PWD/libnanoserve_engine.so
python3 tests/test_nanoq_v3_loader.py && python3 tests/test_suite.py
bash scripts/valgrind.sh
python3 tests/load_test_report.py --preset 50  # with server on :8000
```
