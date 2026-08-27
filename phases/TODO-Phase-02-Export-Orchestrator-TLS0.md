# TODO Phase 02 — Export, Orchestrator, TLS-0

> **Copy-paste this file into Agent mode to implement Phase 02.**
>
> **Master plan:** [TODO-NanoServe-Industry-grade-plan.md](../TODO-NanoServe-Industry-grade-plan.md) — Part A Phase 3+4, Part C TLS-0
> **Prerequisite:** [TODO-Phase-01-Native-Foundation.md](TODO-Phase-01-Native-Foundation.md) Human checkpoint PASS  
> **Agent handoff (read first):** [TODO-Phase-01-Handoff.md](TODO-Phase-01-Handoff.md)  
> **Index:** [TODO-Phase-INDEX.md](TODO-Phase-INDEX.md)

---

## Goal

Ship **HF → `.nanoq` v3 export**, orchestrator wiring (streaming optional), and **TLS-0** (chunk index + chunked forward parity vs full load) so models can be downloaded from the Web UI and layer streaming correctness is proven.

---

## Prerequisites

- Phase 01 complete: native transformer graph generates real text
- distilgpt2-class forward works on CPU

---

## Scope

### Phase 3 — Quantizer & export pipeline

```bash
nanoserve-quantizer export hf:distilgpt2 \
  --out models/distilgpt2-int8.nanoq \
  --arch gpt2 --precision int8
```

Export steps: load HF safetensors → map tensor names → quantize → embed tokenizer → write v3 + Rust validation.

`prepare_model()`: safetensors dir → export v3 (not single-matrix v2).

### Phase 4 — Orchestrator integration (zero API break)

| Layer | Change |
|-------|--------|
| `nanoserve/engine/router.py` | None — `format=nanoq` already routes native |
| `nanoserve/engine/worker.py` | Bind streaming FFI if added |
| `nanoserve/models/registry.py` | Store `arch`, `vocab_size`, `quantized`, v3 paths |
| `server/main.py` | Optional SSE via `engine_infer_stream` |
| `server/static/app.js` | Stream tokens when available |

Ensure `engine_reload_model` resets KV cache.

### Phase TLS-0 — Chunk index + correctness

**`.nanoq` v3 index extensions (backward compatible):**

| Field | Purpose |
|-------|---------|
| `chunk_id` | TLS load unit (default: 1 layer per chunk) |
| `layer_idx` | Transformer block index |
| `stage` | `resident` \| `streamed` |
| `prefetch_hint` | mmap `MADV_SEQUENTIAL` order |
| `chunk_hash` | Optional Blake3 per chunk |

**C++ components:**

| Component | Location |
|-----------|----------|
| `LayerStreamScheduler` | `engine/include/layer_stream.hpp` |
| `WeightChunkArena` | `engine/src/layer_stream.cpp` |
| `ActivationRing` | `engine/include/activation_buffer.hpp` |
| `PrefetchWorker` | `engine/src/prefetch.cpp` (stub ok in TLS-0; full in Phase 03) |

Chunked forward must match full-model logits (ε tolerance) for distilgpt2 fixture.

---

## Implementation steps

1. Add `nanoserve/quantizer/export_hf.py`; extend `quantize.py` and `pipeline.py`
2. Wire Web UI download → v3 export path
3. Extend `nanoq_archive.hpp` + Rust `validate.rs` for chunk fields
4. Implement `LayerStreamScheduler` — load chunk → forward layers → unload
5. Export writes `chunk_id` grouping; validator checks contiguity
6. Add `tests/test_tls_parity.py`
7. Optional: `engine_infer_stream` + UI token streaming
8. Run `scripts/audit_deployments.sh` on native path with v3 model

---

## Files to add/modify

**New:** `nanoserve/quantizer/export_hf.py`, `layer_stream.*`, `activation_buffer.*`, `tests/test_tls_parity.py`, `tests/test_engine_stream.py`

**Modify:** `nanoq_archive.hpp`, `engine_core.cpp`, `registry.py`, `worker.py`, `server/main.py`, `app.js`, `rust/nanoq_runtime/src/validate.rs`

---

## Automated verification

> **Post-build gate:** After **every** Phase 02 build, run **all four** subsections below before the human checkpoint. Do not start the next phase until every row in the verification matrix passes.

### 1. Unit & integration tests

```bash
nanoserve-quantizer export hf:distilgpt2 \
  --out models/distilgpt2-int8.nanoq --arch gpt2 --precision int8

python3 tests/test_tls_parity.py
python3 tests/test_nanoq_v3_loader.py
python3 tests/test_nanoq_vs_gguf.py    # skip if no gguf
python3 tests/test_engine_stream.py
python3 tests/test_suite.py
python3 tests/test_gguf.py

bash scripts/audit_deployments.sh

curl -X POST localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":32,"format":"nanoq","model":"distilgpt2-int8"}'
```

### 2. Performance benchmarks

```bash
# Export throughput (HF → v3 .nanoq)
/usr/bin/time -f 'export_wall=%e maxrss=%M KB' \
  nanoserve-quantizer export hf:distilgpt2 \
    --out /tmp/bench-export.nanoq --arch gpt2 --precision int8

# First-token + full completion latency (native vs GGUF if available)
python3 tests/test_nanoq_vs_gguf.py -v
# Document in documentation/reports/PHASE02_BENCH.md
# Pass: TLS parity logits Δ < ε; export completes without OOM
```

### 3. Memory leak & RSS audits

```bash
./scripts/valgrind.sh
python3 tests/memory_rss_audit.py
python3 tests/memory_server_audit.py || true
python3 tests/test_engine_stream.py   # streaming must not leak handles
```

### 4. Load & stress tests

```bash
python3 tests/load_test_report.py --preset 50 --device cpu --out documentation/reports/PHASE02_LOAD.json
python3 tests/load_test_report.py --preset 50 --device auto --rounds 2 || true
# Pass: native format completions succeed under 50 concurrent users

bash scripts/audit_deployments.sh
```

### Post-build verification matrix

| Category | Command / artifact | Pass criteria |
|----------|-------------------|---------------|
| Unit / integration | `test_tls_parity.py, test_engine_stream.py, audit_deployments.sh` | All PASS |
| Performance | `export + test_nanoq_vs_gguf.py` | Documented; parity within ε |
| Memory leak / RSS | `valgrind.sh, memory_rss_audit.py` | Clean valgrind; RSS stable |
| Load / stress | `load_test_report.py --preset 50` | ≥98% success on format=nanoq |

**Sign-off:** Record results in `documentation/reports/PHASE02_VERIFY.md` (create if missing). CI must run sections 1–4 on every phase merge.

---

## Human checkpoint

| # | What you do | What you should see |
|---|-------------|---------------------|
| 1 | Web UI → Download model (HF distilgpt2) | `.nanoq` v3 file in models dir; registry entry |
| 2 | Web UI → Generate with Native format | Real coherent text (not demo vocab) |
| 3 | Run TLS parity test | Chunked forward == full load (test PASS) |
| 4 | `bash scripts/audit_deployments.sh` | Native path passes with v3 model |
| 5 | Optional: streaming enabled | Tokens appear incrementally in UI |
| 6 | Compare same prompt on `:8002` GGUF | Qualitatively similar English (not identical) |

---

## Acceptance checklist

- [ ] **Post-build gate:** unit/integration + performance + memory leak/RSS + load/stress (see Automated verification); `PHASE02_VERIFY.md` recorded
- [ ] Web UI "Download model" produces v3 `.nanoq`
- [ ] `/v1/completions` with `format=nanoq` returns real text
- [ ] `tests/test_tls_parity.py` passes (logits Δ < ε)
- [ ] `bash scripts/audit_deployments.sh` passes native path
- [ ] Existing API/SDK/TUI work without client changes
- [ ] GGUF `:8002` profile unchanged
- [ ] Legacy v2 demo still loads

---

## Do not break

- GGUF path; v2 demo; default pip install
- `InferenceRouter` API surface

---

## Next phase

[TODO-Phase-03-TLS-Prefetch-Efficiency.md](TODO-Phase-03-TLS-Prefetch-Efficiency.md)

Parallel track may start: [TODO-Phase-04-NMDP-Sandbox.md](TODO-Phase-04-NMDP-Sandbox.md)
---

## Appendix — Phases 3–4 + TLS-0 (full spec)

<a id="part-c-tls-implementation-deep-dive"></a>

# PART C — TLS implementation deep-dive



> Verbatim from [TODO-NanoServe-Industry-grade-plan.md](../TODO-NanoServe-Industry-grade-plan.md)

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

## Phase TLS-0 — Chunk index + correctness

**Goal:** Chunked forward pass produces logits identical (within ε) to full-model load.

### `.nanoq` v3 index extensions (backward compatible)

Add optional fields to index entries (Phase 0 spec):

| Field | Purpose |
|-------|---------|
| `chunk_id` | Group tensors by TLS load unit (default: 1 layer per chunk) |
| `layer_idx` | Transformer block index |
| `stage` | `resident` \| `streamed` (embed/head always `resident`) |
| `prefetch_hint` | Sequential read order for mmap `MADV_SEQUENTIAL` |
| `chunk_hash` | Optional Blake3 sub-hash per chunk for tamper detection |

Export pipeline (`nanoserve/quantizer/export_hf.py`) writes tensors grouped by `chunk_id`; Rust validator (`rust/nanoq_runtime/`) verifies chunk contiguity and bounds.

### C++ components

| Component | Location | Role |
|-----------|----------|------|
| `LayerStreamScheduler` | `engine/include/layer_stream.hpp` | Chunk load/unload orchestration |
| `WeightChunkArena` | `engine/src/layer_stream.cpp` | Buddy-pool slab for current chunk weights |
| `ActivationRing` | `engine/include/activation_buffer.hpp` | In-place hidden-state reuse |
| `PrefetchWorker` | `engine/src/prefetch.cpp` | Background mmap next chunk (`std::thread`) |

### Inference loop (pseudocode)

```cpp
// Per autoregressive token step:
Tensor x = embed(last_token);
for (chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
    WeightChunk chunk = scheduler.load_chunk(chunk_id);  // mmap slice → buddy arena
    for (layer in chunk.layers) {
        x = graph.forward_layer(x, layer, kv_cache[layer_idx], seq_pos);
        // x overwrites prior activation buffer — no growth
    }
    scheduler.unload_chunk(chunk_id);  // release buddy blocks
}
logits = head(norm(x));
```

**Acceptance:** `tests/test_tls_parity.py` — chunked forward == full load (logits Δ < ε) for distilgpt2 fixture.

---
