# Phase 01 Handoff — Agent reference for Phase 02+

> **Read this before starting Phase 02 or any later phase.**
>
> **Prerequisite sign-off:** [documentation/reports/PHASE01_VERIFY.md](../documentation/reports/PHASE01_VERIFY.md)  
> **Deploy & test:** [documentation/Phase-1-Deploy-and-Test.md](../documentation/Phase-1-Deploy-and-Test.md)  
> **Full audit:** [documentation/reports/Phase-1-Test-Report.md](../documentation/reports/Phase-1-Test-Report.md)  
> **Index:** [TODO-Phase-INDEX.md](TODO-Phase-INDEX.md)

---

## Phase 01 status (2026-08-27)

| Item | Status |
|------|--------|
| `.nanoq` v3 archive + Blake3 footer | ✅ Shipped |
| Rust `nanoq_runtime` (validate + GPT-2 BPE tokenizer) | ✅ Shipped |
| GPT-2 transformer graph + KV + greedy sampler | ✅ Shipped (`transformer_gpt2.cpp`) |
| `format=nanoq` real LLM text (distilgpt2-class) | ✅ Shipped |
| Legacy v2 GEMV demo | ✅ Preserved (`legacy_demo: true`) |
| GGUF path | ✅ Unchanged |
| Post-build gate | ✅ PASS |

---

## Agent rules (all future phases)

1. **One phase per session** — do not start Phase N+1 until Phase N human checkpoint passes.
2. **Run the 4-part post-build gate** every phase (unit, bench, memory/valgrind, load) → write `documentation/reports/PHASE##_VERIFY.md`.
3. **Re-run Phase 01 regression tests** after every engine/archive change:
   ```bash
   python3 tests/test_nanoq_v3_loader.py
   python3 tests/test_gguf.py
   ./engine/build/test_transformer_gpt2 tests/fixtures/distilgpt2-int8.nanoq
   python3 tests/test_suite.py
   ```
4. **Do not break:** GGUF path, `/v1/completions` API contract, v2 legacy load, default `pip install` size (no mandatory `[gguf]`).

---

## Critical lessons from Phase 01 (avoid regressions)

### Memory / correctness

| Issue | What happened | Rule for next phases |
|-------|---------------|----------------------|
| **OOB past wpe/KV limit** | Sequences >1024 caused `free(): invalid size` (heap corruption) | Always clamp `max_pos` to `min(config.max_seq_len, wpe.rows)`; bounds-check every forward path including TLS chunked forward |
| **KV not reset between prompts** | Stale KV leaked context across requests | Call `reset_kv()` before each infer; same on `engine_reload_model` and chunk unload |
| **Residual double-add in layer norm** | Wrong logits / garbled text | When touching `transformer_block`, verify residual wiring matches HF GPT-2 |

### Weight layout (GPT-2 export)

- HF Conv1D weights are **`[in, out]`**; engine expects **`[out, in]`** for matmul — transpose on export except `wte`, `wpe`, `lm_head`.
- **`lm_head` transpose bug** in Phase 01 export caused wrong logits — golden test: `best_token=383` for `"Hello"`.
- Phase 02 export pipeline must preserve this mapping; add parity test vs HuggingFace logits.

### Rust ↔ C++ archive sync

- v3 index changes must update **both** `engine/src/nanoq_archive.cpp` **and** `rust/nanoq_runtime/src/validate.rs` (+ `archive.rs`).
- Blake3 footer covers all bytes before footer — any index extension must stay backward compatible.

### Demo / CI reality

- **`*.nanoq` is gitignored** — clones have no model until export/fixture build.
- Phase 02 must ship **`nanoserve-quantizer export hf:…`** as the normal path (not only `tests/fixtures/build_distilgpt2_v3_fixture.py`).
- Document human demo steps in `documentation/Phase-1-Deploy-and-Test.md` (update per phase).

---

## Known deferrals (intentionally not in Phase 01)

| Item | Location today | Target phase |
|------|----------------|--------------|
| Separate `kv_cache.*`, `sampler.cpp`, `engine/src/ops/` | Inline in `transformer.hpp` / `transformer_gpt2.cpp` | Refactor optional in 02+ |
| `gemm_fp16`, `gemm_fp4`, `attention_qkv` | Default stubs in `backend.hpp` | Phase 02–03 perf |
| Llama SentencePiece tokenizer | GPT-2 BPE only in Rust | Phase 02+ |
| `engine_infer_stream`, `engine_set_sampler` | Not implemented | Phase 02 orchestrator |
| Full `third_party/llama.cpp` kernels | CMake stub | Phase 02+ |
| True int8 distilgpt2 quality | Fixture defaults to **fp32** (~486 MB) | Phase 02 export |

Do not half-implement deferrals — either ship fully with tests or leave documented stubs.

---

## Phase 02 focus (next agent session)

Copy **[TODO-Phase-02-Export-Orchestrator-TLS0.md](TODO-Phase-02-Export-Orchestrator-TLS0.md)** into Agent mode **with this handoff file**.

| Priority | Deliverable |
|----------|-------------|
| 1 | `nanoserve/quantizer/export_hf.py` — HF → v3 with correct tensor names/layout |
| 2 | int8 export quality — compare logits/text vs fp32; don't ship broken quant |
| 3 | TLS-0 — chunk index + **parity test** (chunked forward == full model, ε tolerance) |
| 4 | Registry + Web UI download → v3 path |
| 5 | Optional `engine_infer_stream` + SSE (only if parity gate passes first) |

**TLS-0 rule:** correctness before memory wins — chunked forward must match full-model logits for distilgpt2 before optimizing prefetch (Phase 03).

---

## Parallel tracks (don't block yourself)

| Track | Can start after | Notes |
|-------|-----------------|-------|
| Native + TLS | Phase 01 ✅ → **02 → 03 → 09** | Core path |
| RAG (04–06) | Phase 02 + 04 | May use **GGUF** for inference while native export matures |
| QLoRA (07) | Phase 02 | GGUF ok for train/infer initially |
| Federated (08) | Phase 04 + 07 | Needs NMDP sandbox |

---

## Key files map (Phase 01 baseline)

| Area | Files |
|------|-------|
| v3 archive | `engine/include/nanoq_archive.hpp`, `engine/src/nanoq_archive.cpp` |
| Loader | `engine/include/nanoq_loader.hpp`, `engine/src/nanoq_loader.cpp` |
| Rust safety | `rust/nanoq_runtime/src/{validate,archive,tokenizer,manifest}.rs` |
| Transformer | `engine/include/transformer.hpp`, `engine/src/transformer_gpt2.cpp` |
| Engine integration | `engine/src/engine_core.cpp`, `engine/src/engine_ffi.cpp` |
| Python worker | `nanoserve/engine/worker.py` |
| Fixture builder | `tests/fixtures/build_distilgpt2_v3_fixture.py` |
| Phase 01 tests | `tests/test_nanoq_v3_loader.py`, `tests/test_tokenizer_rust.py`, `engine/tests/test_transformer_gpt2.cpp` |

---

## Build commands (baseline)

```bash
cd allocator && cargo build --release && cd ..
cd rust/nanoq_runtime && cargo build --release && cd ../..
cd engine/build && cmake .. -DNANOSERVE_LLAMA_CPP_KERNELS=1 && make -j$(nproc) && cd ../..

export LD_LIBRARY_PATH=$PWD/allocator/target/release:$LD_LIBRARY_PATH
export NANOSERVE_ENGINE_LIB=$PWD/engine/build/libnanoserve_engine.so
export PYTHONPATH=$PWD:$PYTHONPATH

# Model (not in git):
pip install blake3 transformers torch
python3 tests/fixtures/build_distilgpt2_v3_fixture.py
```

---

## Human checkpoint reminder (Phase 01 — already passed)

```bash
curl -X POST localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":32,"format":"nanoq","model":"tests/fixtures/distilgpt2-int8.nanoq"}'
```

Expect real English, **not** `"the model is fast and efficient…"` demo vocab.

---

## Sign-off

- Phase 01: **APPROVED** — see [Phase-1-Test-Report.md](../documentation/reports/Phase-1-Test-Report.md)
- Next: [TODO-Phase-02-Export-Orchestrator-TLS0.md](TODO-Phase-02-Export-Orchestrator-TLS0.md)
