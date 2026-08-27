# TODO Phase 10 — Platform Stretch (Future)

> **Copy-paste this file into Agent mode to implement Phase 10.**
>
> **Status:** **Future / stretch** — implement only after Phase 02 stable. Much of this phase may be **documentation-only** in v1.
>
> **Master plan:** [TODO-NanoServe-Industry-grade-plan.md](../TODO-NanoServe-Industry-grade-plan.md) — Part A Phase 6 + 7
> **Prerequisite:** [TODO-Phase-02-Export-Orchestrator-TLS0.md](TODO-Phase-02-Export-Orchestrator-TLS0.md) Human checkpoint PASS
> **Index:** [TODO-Phase-INDEX.md](TODO-Phase-INDEX.md)

---

## Goal

Address **WASM browser tier** with tiny v3 models and document **distributed pipeline parallelism** (Phase 7) for multi-host layer split — without breaking production Docker/native paths.

---

## Prerequisites

- Native v3 inference stable (Phase 02 minimum)
- Phase 03 recommended for TLS-aware WASM caps

---

## Scope

### Phase 6 — WASM tier (stretch)

After native v3 stable:

- Raise `deployment/wasm/nanoserve.js` cap selectively (e.g. 128 MB) with user warning
- Ship **tiny** v3 models only (distilgpt2-int8 if fits)
- Streaming via JS callback from `engine_infer_stream`
- Keep GGUF out of WASM
- See also [TODO-WASM-LEAN.md](../TODO-WASM-LEAN.md), [TODO-RUST_ALLOC-WASM.md](../TODO-RUST_ALLOC-WASM.md)

**WASM build path:** `./scripts/build_wasm.sh` → `deployment/wasm/`

### Phase 7 — Distributed orchestrator (future — design first)

Only after single-node v3 production-ready:

| Feature | Description |
|---------|-------------|
| Pipeline parallelism | Split layers across mesh hosts; coordinator in FastAPI |
| Tensor shards in v3 | Index marks shard id; partial forward RPC |
| Federated registry | Sync model manifests across hosts |
| Format routing | Router picks host with model loaded + lowest queue |

**Not in initial implementation scope** — deliver architecture doc + RPC sketch.

**TLS synergy:** TLS on each host reduces per-node RAM; Phase 7 splits layers across nodes (space vs time).

---

## Implementation steps

### WASM (if implementing)

1. Extend Emscripten build for v3 loader (minimal graph or demo-only)
2. Update `deployment/wasm/nanoserve.js` memory cap with warning UI
3. Wire `engine_infer_stream` callback to browser UI
4. Extend `tests/test_wasm_native.py`, `tests/test_wasm.py`
5. Update `documentation/WASM.md`

### Distributed (documentation minimum)

1. Write `documentation/Distributed-Pipeline.md` — RPC protocol sketch, shard index fields, coordinator flow
2. Extend `.nanoq` v3 index spec with optional `shard_id` field (spec only)
3. Add design mermaid to doc; link from industry plan Phase 7
4. No cluster protocol in production FastAPI until explicit approval

---

## Files to add/modify

**WASM:** `deployment/wasm/`, `scripts/build_wasm.sh`, `documentation/WASM.md`, `tests/test_wasm*.py`

**Distributed doc:** `documentation/Distributed-Pipeline.md` (new)

**Optional spec:** comment or appendix in `engine/include/nanoq_archive.hpp` for `shard_id`

---

## Automated verification

> **Post-build gate:** After **every** Phase 10 build, run **all four** subsections below before the human checkpoint. Do not start the next phase until every row in the verification matrix passes.

### 1. Unit & integration tests

```bash
# WASM (when emcc available)
./scripts/build_wasm.sh
python3 tests/test_wasm_native.py
python3 tests/test_wasm.py

# Full regression stack
python3 tests/test_tls_parity.py || true
python3 tests/test_suite.py
python3 tests/test_gguf.py
bash scripts/audit_deployments.sh
```

### 2. Performance benchmarks

```bash
# WASM infer smoke timing (when built)
/usr/bin/time -f 'wasm_wall=%e maxrss=%M KB' python3 tests/test_wasm_native.py || true
# Native regression benchmark
python3 tests/load_test_report.py --preset 50 --device cpu --out documentation/reports/PHASE10_BENCH_LOAD.json
# Document in documentation/reports/PHASE10_BENCH.md
# Pass: production Docker/native unchanged; WASM opt-in only
```

### 3. Memory leak & RSS audits

```bash
./scripts/valgrind.sh
python3 tests/test_wasm_native.py   # WASM linear memory cap warnings
python3 tests/memory_rss_audit.py
python3 tests/memory_concurrent_audit.py
# Pass: no native engine regression from WASM build flags
```

### 4. Load & stress tests

```bash
python3 tests/load_test_report.py --preset 50 --device cpu --out documentation/reports/PHASE10_LOAD.json
python3 tests/load_test_report.py --preset 300 --device cpu --out documentation/reports/PHASE10_LOAD_300.json || true
# Full production stress preset (document in STRESS_REPORT lineage)
# Pass: ≥98% at 50; 300-user run recorded for release sign-off

bash scripts/audit_deployments.sh
python3 tests/test_suite.py
```

### Post-build verification matrix

| Category | Command / artifact | Pass criteria |
|----------|-------------------|---------------|
| Unit / integration | `test_wasm_native.py, test_wasm.py, audit_deployments.sh` | WASM smoke or deferral noted |
| Performance | `wasm + native load benchmark` | Documented; native unchanged |
| Memory leak / RSS | `valgrind.sh + memory audits` | Clean valgrind; WASM cap respected |
| Load / stress | `load_test_report.py --preset 50 (300 release)` | ≥98% at 50; 300 recorded |

**Sign-off:** Record results in `documentation/reports/PHASE10_VERIFY.md` (create if missing). CI must run sections 1–4 on every phase merge.

---

## Human checkpoint

| # | What you do | What you should see |
|---|-------------|---------------------|
| 1 | `npx serve deployment/wasm` + load tiny v3 `.nanoq` | Generate works in browser OR documented "deferred" with reason |
| 2 | WASM memory cap warning | User warned above safe model size |
| 3 | Read `documentation/Distributed-Pipeline.md` | Clear Phase 7 design: pipeline RPC, shard index, coordinator |
| 4 | Production Docker/native | **Unchanged** — WASM is opt-in fourth tier |
| 5 | GGUF | Still not in WASM bundle |

**Acceptable v1 outcome:** Distributed = design doc only; WASM = smoke pass OR documented deferral.

---

## Acceptance checklist

- [ ] **Post-build gate:** unit/integration + performance + memory leak/RSS + load/stress (see Automated verification); `PHASE10_VERIFY.md` recorded
- [ ] WASM: tiny v3 smoke OR written deferral in `documentation/WASM.md`
- [ ] Distributed: architecture doc complete (Phase 7 design)
- [ ] No RAG/train in browser tier
- [ ] `audit_deployments.sh` passes for CPU/GPU/GGUF profiles
- [ ] No breaking changes to `:8000` / `:8002` production paths
- [ ] GGUF remains out of WASM

---

## Do not break

- Docker and native install as production paths
- 300-user scaling tier (`run_native_300.sh`)
- RAG/train from Phases 04–08 (not in WASM)

---

## Program complete

When Phases 01–08 pass and 09–10 are signed off (implemented or explicitly deferred), the [industry plan](../TODO-NanoServe-Industry-grade-plan.md) unified acceptance checklist should be re-verified end-to-end.

**Optional follow-up:** [Future-Scope-Rust-port-for-C++Part.md](../Future-Scope-Rust-port-for-C++Part.md) after Phase 03.

---

## Sign-off

```markdown
## Phase 10 sign-off
- WASM: IMPLEMENTED / DEFERRED
- Distributed design doc: DONE
- Date:
- Notes:
```
---

## Appendix — Phases 6–7 + resource budgets (full spec)

> Verbatim from [TODO-NanoServe-Industry-grade-plan.md](../TODO-NanoServe-Industry-grade-plan.md)

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

## Resource budgets (reference targets)

| Component | distilgpt2-class KB (~10k chunks) |
|-----------|-----------------------------------|
| int8 vectors (384-d) | ~4 MB vectors + ~2 MB HNSW graph |
| Chunk store | ~20–50 MB (deduped text) |
| Hot chunk cache | 64–256 MB configurable |
| Embed model (gte-small Q2) | ~25 MB mmap |
| LLM (separate slot) | per existing GGUF/native budgets |
| Staging shard (train job) | deleted after job; cap `NANOSERVE_NMDP_MAX_BYTES_PER_JOB` |
| Sessions | `NANOSERVE_MAX_SESSIONS` × ~1 MB metadata avg |

---
