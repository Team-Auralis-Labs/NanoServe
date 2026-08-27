# TODO Phase 07 — Train Adapter + QLoRA

> **Copy-paste this file into Agent mode to implement Phase 07.**
>
> **Master plan:** [TODO-NanoServe-Industry-grade-plan.md](../TODO-NanoServe-Industry-grade-plan.md) — Part B: T0 + T1
> **Prerequisite:** [TODO-Phase-02-Export-Orchestrator-TLS0.md](TODO-Phase-02-Export-Orchestrator-TLS0.md) Human checkpoint PASS (GGUF path ok)
> **Index:** [TODO-Phase-INDEX.md](TODO-Phase-INDEX.md)

---

## Goal

Ship **`.nanoadapt` adapter format**, registry extension, and **local QLoRA trainer** so a JSONL fixture trains an adapter and inference with `adapter_id` produces measurably different output vs base model.

---

## Prerequisites

- Inference working (GGUF recommended for T1)
- Phase 04 not required for local-only train (needed for Phase 08 federated)

---

## Scope

### Phase T0 — Adapter format + registry

**`.nanoadapt` pack:**

```
[4B magic: 0x4E414453]   # "NADS"
[4B index_len]
[index JSON: lora tensors + offsets]
[config JSON: rank, alpha, target_modules, base_model_id]
[tensor payloads]
[32B Blake3 footer]
```

Extend `ModelEntry.extra` in `nanoserve/models/registry.py` with `adapters[]`.

`InferenceRouter.submit(..., adapter_id=...)` → load base + adapter (merged export ok for GGUF v1).

### Phase T1 — Local QLoRA trainer

**Optional extra:**

```toml
train = ["peft>=0.10", "bitsandbytes>=0.43", "datasets>=2.14", "accelerate>=0.27"]
```

**Module:** `nanoserve/train/qlora.py`

- Input: JSONL `{ "prompt", "completion" }` or `{ "text" }`
- Output: `.nanoadapt` + registry entry
- CPU: rank ≤ 4, tiny datasets; GPU: distilgpt2-class default

**API:**

```python
POST /v1/train/jobs
{
  "base_model": "distilgpt2-Q2_K",
  "adapter_id": "support-bot-v1",
  "data_job_id": null,
  "config": { "rank": 8, "alpha": 16, "epochs": 3, "lr": 2e-4 }
}
GET /v1/train/jobs/{id}
POST /v1/train/adapters/register
```

### Environment variables

```bash
NANOSERVE_ENABLE_TRAIN=0
NANOSERVE_ADAPTERS_DIR=~/.nanoserve/adapters
NANOSERVE_TRAIN_DEFAULT_RANK=8
NANOSERVE_TRAIN_DEFAULT_EPOCHS=3
NANOSERVE_STAGING_DIR=~/.nanoserve/staging
```

---

## Implementation steps

1. Define `.nanoadapt` writer/reader spec in `nanoserve/train/spec.py`
2. Extend registry with adapter manifests
3. Implement `nanoserve/train/qlora.py` job coordinator
4. Add `server/train_routes.py`; mount when `NANOSERVE_ENABLE_TRAIN=1`
5. Wire `adapter_id` through `GenerateRequest` and `InferenceRouter` (stub → full)
6. Add `[train]` optional extra to `pyproject.toml`
7. Add `tests/test_train_qlora.py`

---

## Files to add/modify

**New:** `nanoserve/train/`, `server/train_routes.py`, `tests/test_train_qlora.py`

**Modify:** `nanoserve/models/registry.py`, `nanoserve/engine/router.py`, `server/main.py`, `pyproject.toml`, `nanoserve/__init__.py`

---

## Automated verification

> **Post-build gate:** After **every** Phase 07 build, run **all four** subsections below before the human checkpoint. Do not start the next phase until every row in the verification matrix passes.

### 1. Unit & integration tests

```bash
pip install -e ".[train]"

export NANOSERVE_ENABLE_TRAIN=1

python3 tests/test_train_qlora.py
python3 tests/test_suite.py   # inference regression

curl -X POST localhost:8000/v1/train/jobs \
  -H 'Content-Type: application/json' \
  -d '{"base_model":"distilgpt2-Q2_K","adapter_id":"bot-v1","config":{"rank":8,"epochs":1}}'

curl -s localhost:8000/v1/train/jobs/JOB_ID | jq .
```

### 2. Performance benchmarks

```bash
# QLoRA smoke job duration on JSONL fixture
/usr/bin/time -f 'train_wall=%e maxrss=%M KB' python3 tests/test_train_qlora.py
# Document wall time + peak RSS in documentation/reports/PHASE07_BENCH.md
# Pass: adapter output ≠ base on held-out prompt; job completes within timeout
```

### 3. Memory leak & RSS audits

```bash
python3 tests/test_train_qlora.py
./scripts/valgrind.sh
python3 tests/memory_rss_audit.py
# Run 3 sequential train jobs — adapter dir must not leak temp checkpoints
for i in 1 2 3; do NANOSERVE_ENABLE_TRAIN=1 python3 tests/test_train_qlora.py; done
# Pass: temp dirs cleaned; RSS returns to baseline after job complete
```

### 4. Load & stress tests

```bash
export NANOSERVE_ENABLE_TRAIN=1
# Inference must stay healthy while train job runs in background
python3 tests/load_test_report.py --preset 50 --device cpu --out documentation/reports/PHASE07_LOAD.json &
LOAD_PID=$!
NANOSERVE_ENABLE_TRAIN=1 python3 tests/test_train_qlora.py
wait $LOAD_PID
# Pass: load test ≥98% success during concurrent train smoke

python3 tests/test_train_qlora.py
```

### Post-build verification matrix

| Category | Command / artifact | Pass criteria |
|----------|-------------------|---------------|
| Unit / integration | `test_train_qlora.py, test_suite.py` | Adapter ≠ base; infer OK |
| Performance | `test_train_qlora.py `/usr/bin/time`` | Duration + RSS documented |
| Memory leak / RSS | `3× train job loop + valgrind.sh` | No temp leak; clean valgrind |
| Load / stress | `load_test_report.py during train` | ≥98% infer success |

**Sign-off:** Record results in `documentation/reports/PHASE07_VERIFY.md` (create if missing). CI must run sections 1–4 on every phase merge.

---

## Human checkpoint

| # | What you do | What you should see |
|---|-------------|---------------------|
| 1 | Start train job on local JSONL fixture | Job status progresses; completes |
| 2 | Inspect `~/.nanoserve/adapters/` | `bot-v1.nanoadapt` artifact |
| 3 | Infer with `adapter_id` vs base same prompt | **Different output** on held-out prompt |
| 4 | Register adapter via API | Appears in registry / model metadata |
| 5 | Train with `NANOSERVE_ENABLE_TRAIN=0` | Routes disabled; inference unchanged |

---

## Acceptance checklist

- [ ] **Post-build gate:** unit/integration + performance + memory leak/RSS + load/stress (see Automated verification); `PHASE07_VERIFY.md` recorded
- [ ] `.nanoadapt` artifact produced from local JSONL
- [ ] `POST /v1/train/jobs` and `GET /v1/train/jobs/{id}` work
- [ ] Inference with `adapter_id` differs from base output
- [ ] API accepts `adapter_id` on completions
- [ ] Default pip install unchanged without `[train]`
- [ ] `/v1/completions` without adapter unchanged
- [ ] GGUF path still works

---

## Do not break

- Full fine-tune as default (LoRA/QLoRA only)
- WASM tier — no training in browser
- Existing inference without adapter flags

---

## Next phase

[TODO-Phase-08-Train-Federated-Deploy.md](TODO-Phase-08-Train-Federated-Deploy.md)

**Note:** TLS-Train native backward is stretch — see [TODO-Phase-09-TLS-Advanced-Stretch.md](TODO-Phase-09-TLS-Advanced-Stretch.md)
---

## Appendix — Phases T0–T1 + training routes/env (full spec)

> Verbatim from [TODO-NanoServe-Industry-grade-plan.md](../TODO-NanoServe-Industry-grade-plan.md)

## Phase T0 — Adapter format + registry

### LoRA adapter pack (`.nanoadapt`)

```
[4B magic: 0x4E414453]   # "NADS"
[4B index_len]
[index JSON: lora tensors + offsets]
[config JSON: rank, alpha, target_modules, base_model_id]
[tensor payloads]
[32B Blake3 footer]
```

Alternatively: embed adapter tensors as a labeled section in `.nanoq` v3 index (post full engine).

### Registry extension

Extend `ModelEntry.extra` in `nanoserve/models/registry.py`:

```json
{
  "adapters": [
    {
      "id": "support-bot-v1",
      "path": "~/.nanoserve/adapters/support-bot-v1.nanoadapt",
      "base_model": "distilgpt2-Q2_K",
      "rank": 8,
      "created_at": "..."
    }
  ]
}
```

### Inference routing

- `InferenceRouter.submit(..., adapter_id="support-bot-v1")` → load base + adapter
- GGUF v1: document as **future** (llama.cpp LoRA); v1 may serve merged export only

**Acceptance:** Register adapter manifest; API accepts `adapter_id` field (stub ok until T1 completes).

---

## Phase T1 — Local QLoRA trainer

### Optional extra

```toml
# pyproject.toml
train = ["peft>=0.10", "bitsandbytes>=0.43", "datasets>=2.14", "accelerate>=0.27"]
```

### Trainer module (`nanoserve/train/qlora.py`)

- Input: JSONL `{ "prompt", "completion" }` or `{ "text" }` in staging shard
- Base: HuggingFace model or GGUF-exported weights path
- Output: `.nanoadapt` pack + registry entry
- CPU: rank ≤ 4, tiny datasets only; GPU profile: distilgpt2-class default

### API

```python
POST /v1/train/jobs
{
  "base_model": "distilgpt2-Q2_K",
  "adapter_id": "support-bot-v1",
  "data_job_id": "uuid",           # NMDP staging shard
  "config": { "rank": 8, "alpha": 16, "epochs": 3, "lr": 2e-4 }
}

GET /v1/train/jobs/{id}
POST /v1/train/adapters/register
```

**Acceptance:** Train adapter on local JSONL fixture; register; inference path accepts adapter_id (merged or sidecar per format support).


> **TLS-Train cross-link (Phase T1):** Local QLoRA via `[train]` extra is the default training path. For edge hosts without PyTorch, **TLS-Train** (native streamed backward + gradient checkpointing) is documented in [Part C](#part-c-tls-implementation-deep-dive) as stretch after T1. Adapter weights stay resident; base weights stream per chunk.


### Training routes

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/train/jobs` | Start QLoRA job |
| GET | `/v1/train/jobs/{id}` | Progress + metrics |
| POST | `/v1/train/adapters/register` | Register completed adapter |

---

## Environment variables

```bash
# RAG
NANOSERVE_ENABLE_RAG=0                    # 1 to enable RAG router
NANOSERVE_CORPORA_DIR=~/.nanoserve/corpora
NANOSERVE_EMBED_MODEL=gte-small.Q2_K.gguf   # GGUF embedder path or id
NANOSERVE_RAG_TOP_K=8
NANOSERVE_RAG_RERANK=0
NANOSERVE_SESSION_TTL_S=3600
NANOSERVE_MAX_SESSIONS=128
NANOSERVE_CORPUS_CHUNK_CACHE_MB=256

# NMDP (mesh data plane)
NANOSERVE_ENABLE_NMDP=0
NANOSERVE_NMDP_PORT=8010
NANOSERVE_NMDP_TLS=1                        # require TLS off localhost
NANOSERVE_NMDP_JOB_TTL_S=7200
NANOSERVE_NMDP_MAX_BYTES_PER_JOB=1073741824
NANOSERVE_NMDP_SECRET=                      # HMAC key for capability tokens

# Training
NANOSERVE_ENABLE_TRAIN=0
NANOSERVE_ADAPTERS_DIR=~/.nanoserve/adapters
NANOSERVE_TRAIN_DEFAULT_RANK=8
NANOSERVE_TRAIN_DEFAULT_EPOCHS=3
NANOSERVE_STAGING_DIR=~/.nanoserve/staging
```
