# TODO-nanoq-full-blown-engine — Implementation Prompt for NanoServe

> **Copy-paste this file into an agent session to implement a full native `.nanoq` LLM runtime.**

---

## Prompt header

You are implementing a **full native `.nanoq` LLM runtime** for **NanoServe** — a minimal LLM orchestrator (Rust buddy allocator, C++23 native engine, FastAPI, Python SDK). The runtime must make **`format=nanoq`** run **real autoregressive transformer inference** — efficient, low-RAM, orchestrator-ready — while **GGUF remains optional and coexisting**.

**Architecture choice (confirmed):**

- **C++23** in `libnanoserve_engine.so` (extend current engine + `.nanoq` v3)
- **Adapt llama.cpp internals** (quant kernels, RoPE, sampling) — vendor subset only; no GGUF loader at runtime
- **Rust** for safety-critical boundaries (archive validation, tokenizer, checksums)

**Tagline:** *Native `.nanoq` = real LLM* — not *GEMV demo shuffle*.

Current stack:

```
Rust buddy_alloc → C++ libnanoserve_engine.so → Python FastAPI (InferenceRouter)
                  ↘ optional GGUF (llama-cpp-python) — unchanged, coexisting
```

---

## Problem statement

Today, native `.nanoq` is a **single-matrix GEMV demo** (`engine/src/engine_core.cpp`): fake activations, 22-word vocab, one scalar dot product. Real LLM output only comes from **GGUF + llama.cpp** (`nanoserve/engine/gguf_worker.py`).

Goal: **`format=nanoq`** produces **coherent LLM text** (distilgpt2-class and beyond) without requiring llama-cpp on the native path.

---

## Non-goals (v1)

- Do **not** remove or break the GGUF path (`format=gguf`, Docker `:8002`).
- Do **not** WASM-compile full LLM in v1 (Phase 6 stretch only).
- Do **not** implement distributed tensor parallelism in v1 (Phase 7 future).
- Do **not** ship GGUF parser as primary weight loader — output format is `.nanoq` v3.
- Do **not** break v2 single-matrix demo load (legacy flag until deprecated).

---

## Design principles

| Principle | Rule |
|-----------|------|
| **Orchestrator-first** | Keep `InferenceRouter` → `EnginePool` → `EngineWorker` → FFI unchanged where possible |
| **Lean native path** | No Python in hot loop; C++23 owns forward pass + sampling |
| **Reuse, don’t reinvent** | Adapt llama.cpp tensor layouts, quant kernels, sampling — vendor subset under `third_party/llama.cpp/` |
| **Rust safety net** | Tokenizer, archive validation, Blake3 checksums, config schema |
| **Resource-constrained** | mmap weights, KV cache in buddy pool, int8/fp4 default, lazy layer load |
| **GGUF coexistence** | `.nanoq` = primary native format; GGUF stays optional for HF/community models |
| **Backward compat** | v2 single-matrix files load as **legacy demo mode** (`legacy_demo: true`) |

---

## Target architecture

```mermaid
flowchart TB
  subgraph clients [Clients]
    API[FastAPI_MicroBatcher]
    SDK[NanoServe_SDK]
    WASM[WASM_tier_stretch]
  end

  Router[InferenceRouter]
  Pool[EnginePool]
  Worker[EngineWorker_ctypes]

  subgraph engine [libnanoserve_engine.so]
    FFI[engine_ffi.cpp]
    Core[engine_core.cpp]
    Graph[TransformerGraph]
    KV[KVCache_buddy_pool]
    Backends[CPU_SIMD_CUDA_OpenCL]
    Loader[nanoq_loader_v3]
  end

  subgraph rust [rust_nanoq_runtime]
    Tok[tokenizer]
    Val[archive_validator]
    Meta[config_schema]
  end

  subgraph tools [Build_tools]
    Quant[nanoserve-quantizer_v3]
    Export[hf_to_nanoq_exporter]
    LlamaRef[llama.cpp_kernel_reference]
  end

  clients --> Router
  Router -->|format_nanoq| Pool --> Worker --> FFI --> Core
  Core --> Graph --> Backends
  Core --> KV
  Loader --> Graph
  rust --> Loader
  tools --> Loader
  WASM -.-> FFI
```

---

## Phase 0 — Specification (`.nanoq` v3)

**New format:** archive container (not single matrix). Keep v2 reader for legacy.

### On-disk layout

```
[4B LE magic: 0x4E515033]   # "NQP3"
[4B LE index_len]
[index JSON: tensor catalog + offsets]
[4B LE config_len]
[config JSON: architecture, hyperparams]
[4B LE tokenizer_len]
[tokenizer blob: BPE/SentencePiece bytes]
[tensor payloads: 64-byte aligned, mmap-friendly]
[32B footer: Blake3 hash of all bytes before footer]
```

### Index entry (per tensor)

| Field | Description |
|-------|-------------|
| `name` | e.g. `blk.0.attn_q.weight` |
| `dtype` | `int8` \| `fp16` \| `fp4` \| `fp32` |
| `shape[]` | Tensor dimensions |
| `offset`, `size` | Byte range in payload section |
| `scale_offset` | Optional scales blob offset |
| `quant` | `none` \| `per-row` \| `per-block` (block_size 32/64/128) |

### Config block (architecture enum)

**v1 targets:** `gpt2`, `llama` (covers distilgpt2, SmolLM-class)

| Field | Example |
|-------|---------|
| `arch` | `"gpt2"` |
| `vocab_size` | 50257 |
| `hidden_size` | 768 |
| `n_layers` | 6 |
| `n_heads` | 12 |
| `n_kv_heads` | 12 (GQA for Llama) |
| `max_seq_len` | 2048 |
| `norm_eps` | 1e-5 |
| `rope_theta` | 10000 |
| `act_fn` | `"gelu"` |

### Files to add/extend

| File | Action |
|------|--------|
| `engine/include/nanoq_loader.hpp` | Add `NanoqArchive`, `NanoqConfig`, v3 API |
| `engine/include/nanoq_archive.hpp` | v3 mmap parser (new) |
| `engine/src/nanoq_loader.cpp` | v3 + v2 legacy fallback |
| `rust/nanoq_runtime/` | Footer hash, index sanity, bounds checks |
| `nanoserve/quantizer/quantize.py` | v3 writer |

**Acceptance:** Load distilgpt2-class fixture `.nanoq` v3; `engine_model_info` returns architecture + layer count.

---

## Phase 1 — Rust safety layer

New crate: `rust/nanoq_runtime/`

| Module | Responsibility |
|--------|----------------|
| `validate` | Parse index/config off hot path; reject OOB offsets; Blake3 footer |
| `tokenizer` | BPE (GPT-2) + SentencePiece (Llama) — encode/decode FFI |
| `manifest` | Model card metadata for registry |

**FFI surface (C ABI from Rust):**

```c
int  nanoq_archive_validate_path(const char* path);
int  nanoq_archive_validate(const uint8_t* data, size_t len);
void* nanoq_tokenizer_create(const uint8_t* data, size_t len);
void  nanoq_tokenizer_destroy(void* handle);
int  nanoq_tokenizer_encode(void* handle, const char* text, uint32_t* out_ids, size_t max_ids);
char* nanoq_tokenizer_decode(void* handle, const uint32_t* ids, size_t num_ids);
void  nanoq_string_free(char* s);
int  nanoq_tokenizer_vocab_size(void* handle);
```

Wire into `engine/src/engine_ffi.cpp`; embed tokenizer handle in `EngineHandle`.

**Acceptance:** Round-trip encode/decode for GPT-2 BPE matches `transformers` on fixture strings.

---

## Phase 2 — C++ transformer graph (core inference)

Replace demo loop in `engine/src/engine_core.cpp`.

### New components

| Component | Location | Notes |
|-----------|----------|-------|
| `TransformerModel` | `engine/include/transformer.hpp` | Owns tensor refs into mmap archive |
| `TransformerGraph` | `engine/src/transformer_gpt2.cpp`, `transformer_llama.cpp` | Architecture-specific forward |
| `KVCache` | `engine/include/kv_cache.hpp` | Per-layer K/V; allocate from buddy pool |
| `Sampler` | `engine/src/sampler.cpp` | greedy, top-k, top-p, temperature |
| `TensorOps` | `engine/src/ops/` | matmul, rms_norm, rope, softmax, silu/gelu |

### llama.cpp adaptation strategy

- **Copy/adapt** (with license header): Q4/Q8 matmul micro-kernels, RoPE, softmax, sampling
- **Do not** depend on GGUF loader at runtime
- `third_party/llama.cpp/` vendor subset — CMake option `NANOSERVE_LLAMA_CPP_KERNELS=1`
- Map `ggml_type` patterns → `.nanoq` v3 dtype enums in thin adapter header

### Backend extensions (`engine/include/backend.hpp`)

Extend beyond scalar GEMV:

- `gemm_int8`, `gemm_fp16`, `gemm_fp4` (M×K × K×N)
- `attention_qkv` fused where SIMD allows
- CUDA/OpenCL: int8/fp16 matmul first; attention on CPU if needed

### Fix existing bug

Apply per-row / per-block scales in int8 paths (`engine/src/backend_cpu.cpp`) — scales loaded but unused today.

### New FFI (backward compatible)

Keep `engine_infer()`; add optional:

```c
int engine_infer_stream(void* handle, const char* prompt, int max_tokens,
                        void (*callback)(const char* token, void* user), void* user);
int engine_reset_kv(void* handle);
int engine_set_sampler(void* handle, const char* json_opts);
```

**Acceptance:** distilgpt2 `.nanoq` v3 generates coherent English on CPU; output ≠ 22-word vocab demo.

---

## Phase 3 — Quantizer & export pipeline

Extend `nanoserve/models/pipeline.py` and quantizer.

### CLI

```bash
# Full model export (safetensors / HF repo)
nanoserve-quantizer export hf:distilgpt2 \
  --out models/distilgpt2-int8.nanoq \
  --arch gpt2 --precision int8

# Single-tensor legacy v2 still supported
nanoserve-quantizer --rows 256 --cols 1024 --out demo.nanoq
```

### Export steps

1. Load HF weights (safetensors preferred; `.bin` via safetensors conversion)
2. Map tensor names → v3 index (GPT-2 / Llama naming tables)
3. Quantize per-tensor (int8 default; fp4 for large layers optional)
4. Embed `tokenizer.json` / `spm.model`
5. Write v3 archive + Rust validation pass

### Auto-quantize path

`prepare_model()`: if safetensors dir → export v3 (not single-matrix v2).

**Acceptance:** Web UI “Download model” produces v3 `.nanoq`; `/v1/completions` with `format=nanoq` returns real text.

---

## Phase 4 — Orchestrator integration (zero API break)

| Layer | Change |
|-------|--------|
| `nanoserve/engine/router.py` | None — `format=nanoq` already routes native |
| `nanoserve/engine/worker.py` | Bind new FFI symbols if streaming added |
| `nanoserve/models/registry.py` | Store `arch`, `vocab_size`, `quantized`, v3 paths |
| `server/main.py` | Optional SSE/streaming via `engine_infer_stream` |
| `server/static/app.js` | Stream tokens when available |

**Multi-model / LRU:** Existing `ModelCache` + per-path workers — ensure `engine_reload_model` resets KV cache.

**Distributed mesh:** No new cluster protocol in v1 — each host runs full v3 model. Nginx `least_conn` unchanged (`documentation/connect-network.md`).

**Acceptance:** `bash scripts/audit_deployments.sh` passes native path with real `.nanoq` v3; distilgpt2 parity vs GGUF on same prompt (qualitative).

---

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

## Phase 6 — WASM tier (stretch)

After native v3 stable:

- Raise `deployment/wasm/nanoserve.js` cap selectively (e.g. 128 MB) with user warning
- Ship **tiny** v3 models only (distilgpt2-int8 if fits)
- Streaming via JS callback from `engine_infer_stream`
- Keep GGUF out of WASM

---

## Phase 7 — Distributed orchestrator extensions (future)

Only after single-node v3 is production-ready:

| Feature | Description |
|---------|-------------|
| **Pipeline parallelism** | Split layers across mesh hosts; coordinator in FastAPI |
| **Tensor shards in v3** | Index marks shard id; partial forward RPC |
| **Federated registry** | Sync model manifests across hosts |
| **Format routing** | Router picks host with model loaded + lowest queue |

Not in initial scope.

---

## Testing strategy

| Test | File |
|------|------|
| v3 loader round-trip | `tests/test_nanoq_v3_loader.py` |
| Tokenizer parity | `tests/test_tokenizer_rust.py` |
| GPT-2 forward golden | `tests/test_transformer_gpt2.cpp` (native) |
| vs GGUF qualitative | `tests/test_nanoq_vs_gguf.py` (skip if no gguf) |
| FFI streaming | `tests/test_engine_stream.py` |
| WASM smoke | extend `tests/test_wasm_native.py` |
| Memory budget | `tests/test_nanoq_memory.py` (RSS cap) |

---

## File touch list (summary)

**New**

- `engine/include/transformer.hpp`, `engine/src/transformer_*.cpp`
- `engine/include/kv_cache.hpp`, `engine/src/kv_cache.cpp`
- `engine/include/nanoq_archive.hpp`
- `engine/src/ops/` (matmul, norm, rope, attn)
- `engine/src/sampler.cpp`
- `rust/nanoq_runtime/` (validate, tokenizer)
- `third_party/llama.cpp/` (vendor subset, CMake gated)
- `nanoserve/quantizer/export_hf.py`
- `tests/test_nanoq_v3_*.py`, `tests/test_transformer_gpt2.cpp`

**Modify**

- `engine/src/engine_core.cpp` — replace demo infer
- `engine/include/nanoq_loader.hpp`, `engine/src/nanoq_loader.cpp`
- `engine/include/backend.hpp`, backend sources
- `engine/src/engine_ffi.cpp`
- `engine/CMakeLists.txt` — Rust crate link, llama.cpp subset
- `nanoserve/quantizer/quantize.py`, `nanoserve/models/pipeline.py`
- `nanoserve/engine/worker.py`
- Docs: `README.md`, `documentation/How-to-add-models-doc.md`, `documentation/WASM.md`

**Do not break**

- GGUF path (`nanoserve/engine/gguf_worker.py`)
- v2 demo `.nanoq` load (legacy flag in `engine_model_info`)

---

## Implementation order (recommended)

1. **Phase 0** — v3 spec + validator (Rust) + empty graph loads
2. **Phase 1** — tokenizer FFI
3. **Phase 2** — GPT-2 graph + KV + sampler (CPU int8)
4. **Phase 3** — HF exporter for distilgpt2
5. **Phase 4** — orchestrator wiring + streaming
6. **Phase 5** — mmap, CUDA matmul, perf tuning
7. **Phase 6** — WASM stretch
8. **Phase 7** — distributed (future)

---

## Test commands

```bash
# Build engine + Rust runtime
cd rust/nanoq_runtime && cargo build --release
cd engine/build && cmake .. && make -j$(nproc)

# Export distilgpt2 to v3
nanoserve-quantizer export hf:distilgpt2 \
  --out models/distilgpt2-int8.nanoq --arch gpt2 --precision int8

# Native inference
curl -X POST localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":32,"format":"nanoq","model":"distilgpt2-int8"}'

# Compare vs GGUF (optional)
curl -X POST localhost:8002/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":32,"format":"gguf","model":"distilgpt2-Q2_K"}'

# Unit tests
python3 tests/test_nanoq_v3_loader.py
python3 tests/test_tokenizer_rust.py
python3 tests/test_nanoq_vs_gguf.py
```

---

## Acceptance checklist

- [ ] `format=nanoq` produces **real LLM text** for distilgpt2-class models
- [ ] **No llama-cpp-python** required on native path
- [ ] GGUF `:8002` profile **unchanged** and coexisting
- [ ] Memory footprint **≤ GGUF Q4** for same model class (mmap + int8)
- [ ] Existing API (`/v1/completions`, SDK, TUI) works without client changes
- [ ] Legacy v2 demo still loads with `legacy_demo: true` in model info
- [ ] Blake3 footer validation rejects tampered archives
- [ ] `engine_reset_kv` clears conversation state between prompts
- [ ] `bash scripts/audit_deployments.sh` passes with v3 model on native path

---

## Success criteria

- Native `.nanoq` is a **first-class LLM runtime**, not a GEMV demo
- Orchestrator (`InferenceRouter`, micro-batcher, mesh HTTP) gains full LLM on `format=nanoq` without API changes
- Resource profile suitable for edge / low-RAM hosts (int8 + mmap + buddy KV)
- Clear migration path from v2 demo → v3 full models
- GGUF remains the compatibility lane for community `.gguf` files
