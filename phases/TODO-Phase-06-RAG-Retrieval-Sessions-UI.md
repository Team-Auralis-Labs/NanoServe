# TODO Phase 06 — RAG Retrieval, Sessions, UI

> **Copy-paste this file into Agent mode to implement Phase 06.**
>
> **Master plan:** [TODO-NanoServe-Industry-grade-plan.md](../TODO-NanoServe-Industry-grade-plan.md) — Part B: R2 + R3 + R4
> **Prerequisite:** [TODO-Phase-05-RAG-Corpus-Ingest.md](TODO-Phase-05-RAG-Corpus-Ingest.md) Human checkpoint PASS
> **Index:** [TODO-Phase-INDEX.md](TODO-Phase-INDEX.md)

---

## Goal

Ship **hybrid retrieval**, **stateful RAG sessions**, **Web UI + SDK + Docker profile** — end-to-end grounded multi-turn chat with chunk citations under bounded memory.

---

## Prerequisites

- Phase 05: corpus ingest + index working
- Inference available (GGUF `:8002` or native from Phase 02)

---

## Scope

### Phase R2 — Retrieval + rerank

| Stage | Implementation |
|-------|----------------|
| Dense | HNSW top-k (Rust) |
| Sparse | BM25 (tantivy or lightweight Rust index) |
| Fusion | RRF or weighted sum |
| Rerank | Optional cross-encoder GGUF; off by default |

**Context budget manager:** fit top-k chunks within LLM `max_context`

**API:**

```python
POST /v1/rag/corpora/{corpus_id}/query
{ "query": "What is the refund policy?", "top_k": 8, "rerank": false }
```

### Phase R3 — Stateful sessions

```bash
NANOSERVE_SESSION_TTL_S=3600
NANOSERVE_MAX_SESSIONS=128
NANOSERVE_SESSION_CHUNK_CACHE_MB=64
NANOSERVE_CORPUS_CHUNK_CACHE_MB=256
```

**Flow:** create session → chat (retrieve + augment + infer) → append messages → TTL/LRU eviction

Call `engine_reset_kv` on session delete when engine handle exists.

### Phase R4 — Deploy integration

- Extend `GET /health`: `rag_available`, `corpora_registered`, `active_sessions`, `nmdp_port`
- Docker profile `rag` on `:8003` + NMDP `:8010`
- Web UI: Corpus panel, Chat mode, mesh job wizard
- SDK: `rag_create_session`, `rag_chat`

**Prompt template:**

```
System: Answer using only the context below. Cite chunk ids when relevant.

Context:
[chunk c_001] ...

User: {message}
```

### RAG routes

| Method | Path |
|--------|------|
| POST | `/v1/rag/corpora/{id}/query` |
| POST | `/v1/rag/sessions` |
| GET | `/v1/rag/sessions/{id}` |
| DELETE | `/v1/rag/sessions/{id}` |
| POST | `/v1/rag/chat` |

---

## Implementation steps

1. Add `nanoserve/rag/retriever.py`, `session.py`, `router.py`
2. Implement hybrid search + context budget manager
3. SessionManager with LRU + TTL; session store on disk
4. Wire `RAGRouter` → `InferenceRouter` with augmented prompt
5. Extend `server/static/app.js` — Corpus + Chat UI
6. Docker compose `rag` profile; extend health endpoint
7. SDK methods in `nanoserve/__init__.py`
8. Add `tests/test_rag_session.py`, `tests/test_rag_chat.py`
9. User guide stub: `documentation/RAG-Retrain.md`

---

## Files to add/modify

**New:** `nanoserve/rag/router.py`, `session.py`, `retriever.py`, `tests/test_rag_session.py`, `tests/test_rag_chat.py`, `documentation/RAG-Retrain.md`

**Modify:** `server/main.py`, `server/static/app.js`, `nanoserve/engine/router.py`, `nanoserve/__init__.py`, `docker-compose.yml`, `documentation/connect-network.md`

---

## Automated verification

> **Post-build gate:** After **every** Phase 06 build, run **all four** subsections below before the human checkpoint. Do not start the next phase until every row in the verification matrix passes.

### 1. Unit & integration tests

```bash
export NANOSERVE_ENABLE_RAG=1
export NANOSERVE_ENABLE_NMDP=1

python3 tests/test_rag_session.py
python3 tests/test_rag_chat.py
python3 tests/test_suite.py   # regression
python3 tests/test_rag_ingest.py   # prior phase regression

# docker compose --profile rag up --build   # optional

curl -X POST localhost:8000/v1/rag/sessions \
  -d '{"corpus_id":"kb1","model":"distilgpt2-Q2_K","format":"gguf"}'

curl -X POST localhost:8000/v1/rag/chat \
  -d '{"session_id":"SESSION_UUID","message":"What is our refund policy?"}'
```

### 2. Performance benchmarks

```bash
# RAG chat p50/p95 latency (fixture corpus, 20 turns)
python3 tests/test_rag_chat.py -v
# Document retrieval + generation latency in documentation/reports/PHASE06_BENCH.md
# Pass: chunk_ids present; p95 within session SLA for fixture KB
```

### 3. Memory leak & RSS audits

```bash
python3 tests/test_rag_session.py   # LRU + TTL under load
python3 tests/memory_server_audit.py || true
python3 tests/memory_rss_audit.py
./scripts/valgrind.sh
# Pass: session count bounded by NANOSERVE_MAX_SESSIONS; eviction works
```

### 4. Load & stress tests

```bash
export NANOSERVE_ENABLE_RAG=1
python3 tests/load_test_report.py --preset 50 --device cpu --out documentation/reports/PHASE06_LOAD.json
python3 tests/load_test_report.py --preset 150 --device cpu --out documentation/reports/PHASE06_LOAD_150.json || true
# Pass: LRU eviction under session cap; ≥98% success at 50 users

python3 tests/test_rag_session.py
python3 tests/test_rag_chat.py
```

### Post-build verification matrix

| Category | Command / artifact | Pass criteria |
|----------|-------------------|---------------|
| Unit / integration | `test_rag_session.py, test_rag_chat.py, test_suite.py` | Citations + LRU PASS |
| Performance | `test_rag_chat.py latency` | p95 documented for fixture KB |
| Memory leak / RSS | `test_rag_session.py + memory audits` | Bounded session memory |
| Load / stress | `load_test_report.py --preset 50 (150 optional)` | ≥98% success |

**Sign-off:** Record results in `documentation/reports/PHASE06_VERIFY.md` (create if missing). CI must run sections 1–4 on every phase merge.

---

## Human checkpoint

| # | What you do | What you should see |
|---|-------------|---------------------|
| 1 | Open Web UI → Corpus panel | Create/ingest corpus visible |
| 2 | Start Chat session | Session picker; model + corpus bound |
| 3 | Multi-turn chat on fixture KB | Grounded replies; **chunk_ids** in response metadata |
| 4 | `GET /health` | `rag_available: true`, session count |
| 5 | Exceed `NANOSERVE_MAX_SESSIONS` in load test | LRU eviction; bounded memory |
| 6 | GGUF `:8002` + RAG `:8003` | Both profiles work independently |
| 7 | SDK `rag_chat` example | Same behavior as API |

---

## Acceptance checklist

- [ ] **Post-build gate:** unit/integration + performance + memory leak/RSS + load/stress (see Automated verification); `PHASE06_VERIFY.md` recorded
- [ ] Hybrid retrieval returns relevant chunks for fixture corpus
- [ ] Stateful multi-turn chat under `NANOSERVE_MAX_SESSIONS` cap
- [ ] Responses include `chunk_ids` citations in metadata
- [ ] Web UI end-to-end RAG chat with citations
- [ ] `/v1/completions` without RAG flags unchanged
- [ ] GGUF `:8002` still works alongside RAG on `:8003`
- [ ] Resource: index + chunk cache within configured MB limits

---

## Do not break

- Stateless completions API (without RAG flags)
- NMDP sandbox from Phase 04
- Default install without `[rag]` extra

---

## Next phase

[TODO-Phase-07-Train-Adapter-QLoRA.md](TODO-Phase-07-Train-Adapter-QLoRA.md)
---

## Appendix — Phases R2–R4 + stateful layer + RAG API/env (full spec)

> Verbatim from [TODO-NanoServe-Industry-grade-plan.md](../TODO-NanoServe-Industry-grade-plan.md)

## Phase R2 — Retrieval + rerank

### Hybrid search

| Stage | Implementation |
|-------|----------------|
| Dense | HNSW top-k (Rust) |
| Sparse | BM25 via `tantivy` or lightweight Rust inverted index |
| Fusion | RRF (reciprocal rank fusion) or weighted sum |
| Rerank | Optional cross-encoder (small GGUF / CPU); off by default |

### Context budget manager

- Given LLM `max_context` (from model config or env), fit top-k chunks without overflow
- Truncate chunks by token estimate; prioritize rerank score
- Return `{ chunk_ids[], context_text, tokens_used }` for injection

### Prompt template (injected before inference)

```
System: Answer using only the context below. Cite chunk ids when relevant.

Context:
[chunk c_001] ...
[chunk c_002] ...

User: {message}
```

### API

```python
POST /v1/rag/corpora/{corpus_id}/query
{
  "query": "What is the refund policy?",
  "top_k": 8,
  "rerank": false
}
```

**Acceptance:** Query returns ranked chunks; context fits configured token budget.

---

## Phase R3 — Stateful RAG sessions

### Session store

```
~/.nanoserve/sessions/
  {session_id}.json       # metadata, message history pointers
  {session_id}.cache      # mmap hot retrieval cache for session
```

### Session record

```json
{
  "session_id": "uuid",
  "corpus_id": "my-kb",
  "model": "distilgpt2-Q2_K",
  "format": "gguf",
  "messages": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "retrieved_chunk_ids": ["c_001", "c_003"],
  "created_at": "...",
  "expires_at": "..."
}
```

### Environment variables

```bash
NANOSERVE_SESSION_TTL_S=3600          # default session lifetime
NANOSERVE_MAX_SESSIONS=128            # LRU cap
NANOSERVE_SESSION_CHUNK_CACHE_MB=64   # per-session retrieval cache
NANOSERVE_CORPUS_CHUNK_CACHE_MB=256   # global hot chunk LRU
```

### Flow

1. `POST /v1/rag/sessions` → create session bound to corpus + model
2. `POST /v1/rag/chat` → retrieve (with session history query expansion optional) → augment prompt → `InferenceRouter.submit()`
3. Append assistant reply; update `retrieved_chunk_ids`
4. Session delete or TTL → free cache; call `engine_reset_kv` if engine session handle exists

### Separation of concerns

| State | Owner |
|-------|-------|
| Message history + retrieval cache | `SessionManager` (Python or Rust) |
| Transformer KV cache | C++ engine (`engine_reset_kv` per [Part A — Phase 2](#phase-2--c-transformer-graph-core-inference)) |

**Acceptance:** Multi-turn chat retrieves relevant chunks; memory bounded under `NANOSERVE_MAX_SESSIONS` load test.

---

## Phase R4 — Deploy integration

### FastAPI / health

Extend `GET /health`:

```json
{
  "rag_available": true,
  "corpora_registered": 2,
  "index_size_bytes": 10485760,
  "active_sessions": 5,
  "nmdp_port": 8010,
  "mesh_jobs_active": 0
}
```

### Docker profile (optional)

```yaml
# docker-compose.yml — profile: rag
nanoserve-rag:
  ports:
    - "8003:8000"
    - "8010:8010"
  environment:
    - NANOSERVE_ENABLE_RAG=1
    - NANOSERVE_NMDP_PORT=8010
  volumes:
    - nanoserve-corpora:/root/.nanoserve/corpora
```

### Web UI additions (`server/static/`)

- **Corpus** panel: create, ingest (local / URL / mesh job)
- **Chat** mode: session picker, grounded replies with chunk citations
- **Mesh job** wizard: generate capability token QR / copy for peer agent

### SDK

```python
from nanoserve import NanoServe

engine = NanoServe()
session = engine.rag_create_session(corpus_id="my-kb", model="distilgpt2-Q2_K", format="gguf")
reply = engine.rag_chat(session_id=session.id, message="Summarize the handbook")
```

**Acceptance:** End-to-end RAG chat from Web UI and SDK on GGUF host; citations visible in response metadata.


## Stateful layer (cross-cutting)

### SessionManager

| Component | Language | Responsibility |
|-----------|----------|----------------|
| `SessionManager` | Python (v1) or Rust (stretch) | CRUD sessions, LRU, TTL |
| `CorpusCache` | Rust | Global chunk hot set |
| `StagingStore` | Python | NMDP job temp files with auto-cleanup |

### Lifecycle hooks

| Event | Action |
|-------|--------|
| Session create | Allocate cache file; bind corpus + model |
| Session chat | Retrieve → infer → append messages |
| Session fork | Copy history; new `session_id`; reset engine KV |
| Session delete | Unlink cache; decrement corpus ref counts |
| Job complete | Delete staging; revoke tokens; audit log finalize |

## API contract (additive)

### Extended completion request

```python
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 24
    device: Literal["cpu", "gpu", "auto"] = "cpu"
    model: Optional[str] = None
    format: Literal["auto", "nanoq", "gguf"] = "auto"
    # RAG extensions (optional)
    session_id: Optional[str] = None
    use_rag: bool = False
    corpus_id: Optional[str] = None
    adapter_id: Optional[str] = None
```

### RAG routes

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/rag/corpora/ingest` | Local or mesh ingest |
| GET | `/v1/rag/corpora` | List corpora |
| POST | `/v1/rag/corpora/{id}/query` | Standalone retrieval |
| POST | `/v1/rag/sessions` | Create stateful session |
| GET | `/v1/rag/sessions/{id}` | Session metadata |
| DELETE | `/v1/rag/sessions/{id}` | End session |
| POST | `/v1/rag/chat` | Session chat (retrieve + infer) |

### Mesh data plane routes

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/mesh/jobs` | Create ingest/train data job |
| GET | `/v1/mesh/jobs/{id}` | Status + audit summary |
| POST | `/v1/mesh/jobs/{id}/cancel` | Revoke tokens, cleanup |
| GET | `/v1/mesh/data/shard/{id}` | Pull shard (capability auth) |

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
