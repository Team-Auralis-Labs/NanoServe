# Phase 01 sign-off

- Date: 2026-08-27
- Automated tests: **PASS**
- Human checkpoint: **PASS**

## Gate 1 — Unit & integration

| Test | Result |
|------|--------|
| `tests/test_nanoq_v3_loader.py` (5 tests) | PASS |
| `tests/test_tokenizer_rust.py` | PASS |
| `engine/build/test_transformer_gpt2` | PASS (best_token=383) |
| `tests/test_nanoq_loader.py` | PASS (v2 legacy) |
| `tests/test_suite.py` | PASS (15/15, 2 skipped) |
| `tests/test_gguf.py` | PASS (8/8; fallback test patched for llama-cpp present) |
| `tests/test_simd_parity.py` | PASS |
| `tests/test_quantizer_fp16_fp4.py` | PASS |
| `tests/test_nanoq_memory.py` | PASS |

## Gate 2 — Performance

| Metric | Value |
|--------|-------|
| Benchmark | See [PHASE01_BENCH.md](PHASE01_BENCH.md) |
| v3 infer (Hello, 32 tok) | ~3.2s CPU, ~993 MB RSS |

## Gate 3 — Memory leak & RSS

| Check | Result |
|-------|--------|
| `tests/memory_rss_audit.py` | PASS (0.0 MiB growth) |
| `tests/memory_concurrent_audit.py` | PASS |
| `scripts/valgrind.sh` | PASS (0 leaks, 1000 cycles) |

## Gate 4 — Load & stress

| Check | Result |
|-------|--------|
| `load_test_report.py --preset 50` | PASS (100%, 50/50) |
| v3 fixture concurrent (10×) | PASS |

## Human checkpoint checklist

1. v3 fixture → `arch=gpt2`, `n_layers=6`, `format=nanoq_v3` — **PASS**
2. Tokenizer round-trip matches `transformers` GPT-2 BPE — **PASS**
3. `curl /v1/completions` `format=nanoq` distilgpt2 — **PASS** (real English, not 22-word GEMV)
4. Legacy v2 loads with `legacy_demo: true` — **PASS**
5. Blake3 footer tamper rejected — **PASS**
6. `engine_reset_kv` isolates prompts — **PASS**

## Known deferrals (non-blocking for Phase 02)

- `kv_cache.*`, `sampler.cpp`, `engine/src/ops/` — logic lives in `transformer.hpp` / `transformer_gpt2.cpp`
- `gemm_fp16`, `gemm_fp4`, `attention_qkv` backend methods — not yet wired (int8 scales fixed)
- Llama SentencePiece tokenizer — GPT-2 BPE only in Rust runtime
- Optional FFI: `engine_infer_stream`, `engine_set_sampler` — Phase 02
- Full `third_party/llama.cpp` kernel subset — CMake stub only
- Fixture exported fp32 (named int8); full int8 quality deferred

## Deliverables shipped

- `.nanoq` v3 archive parser + v2 legacy fallback
- `rust/nanoq_runtime` (Blake3 validate, GPT-2 tokenizer FFI)
- GPT-2 transformer graph + KV + sampler
- Engine v3 path + `engine_reset_kv` FFI
- int8 per-row scale fix (CPU/CUDA/OpenCL)
- Fixture builder + distilgpt2 v3 fixture (~486 MB)

- Signed: Phase 01 audit agent, 2026-08-27
