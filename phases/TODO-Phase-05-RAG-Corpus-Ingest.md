# TODO Phase 05 — RAG Corpus + Ingest

> **Copy-paste this file into Agent mode to implement Phase 05.**
>
> **Master plan:** [TODO-NanoServe-Industry-grade-plan.md](../TODO-NanoServe-Industry-grade-plan.md) — Part B: R0 + R1
> **Prerequisite:** [TODO-Phase-04-NMDP-Sandbox.md](TODO-Phase-04-NMDP-Sandbox.md) Human checkpoint PASS
> **Index:** [TODO-Phase-INDEX.md](TODO-Phase-INDEX.md)

---

## Goal

Ship **corpus specification**, **Rust vector index + chunk store**, and **ingest pipeline** (local + NMDP mesh) with Blake3 dedup — so a fixture corpus indexes and returns top-k chunk ids on query.

---

## Prerequisites

- Phase 04: NMDP sandbox working (for mesh ingest path)
- GGUF embedder available for Phase 05 (`[gguf]` extra + gte-small or similar) OR stub embed path documented

---

## Scope

### Phase R0 — Corpus + index spec

**On-disk layout:**

```
~/.nanoserve/corpora/{corpus_id}/
  manifest.json
  index.hnsw
  index.meta.json
  chunks/{blake3_hex}.chunk
```

**Vector index (`rust/nanoq_rag/`):** HNSW default; int8 embeddings; cosine metric

**Embedding:** small GGUF embedder via existing GGUF path (e.g. gte-small)

### Phase R1 — Ingest pipeline

**Chunkers:** plain text/markdown, JSONL; PDF optional via `[rag]` extra

**Flow:** Source → Chunk → Blake3 dedup → Embed batch → HNSW insert → Update manifest

**API:**

```python
POST /v1/rag/corpora/ingest
{
  "corpus_id": "my-kb",
  "sources": [
    { "type": "local", "path": "/path/to/docs" },
    { "type": "mesh", "job_id": "uuid", "peer_device_ids": ["laptop-alice"] }
  ],
  "chunk_size": 512,
  "chunk_overlap": 64
}
```

### Environment variables

```bash
NANOSERVE_ENABLE_RAG=0
NANOSERVE_CORPORA_DIR=~/.nanoserve/corpora
NANOSERVE_EMBED_MODEL=gte-small.Q2_K.gguf
```

---

## Implementation steps

1. Create `rust/nanoq_rag/` — HNSW, chunk_store, Blake3 addressing
2. Add `nanoserve/rag/spec.py`, `nanoserve/rag/ingest.py`
3. Implement local chunker + embed batching with backpressure
4. Wire NMDP pull → staging → ingest (mesh source type)
5. Add `POST /v1/rag/corpora/ingest`, `GET /v1/rag/corpora`
6. Add `tests/test_rag_ingest.py`
7. Optional `[rag]` extra in `pyproject.toml`

---

## Files to add/modify

**New:** `rust/nanoq_rag/`, `nanoserve/rag/`, `server/rag_routes.py` (ingest + list), `tests/test_rag_ingest.py`

**Modify:** `server/main.py`, `pyproject.toml` (`[rag]` optional)

---

## Automated verification

> **Post-build gate:** After **every** Phase 05 build, run **all four** subsections below before the human checkpoint. Do not start the next phase until every row in the verification matrix passes.

### 1. Unit & integration tests

```bash
export NANOSERVE_ENABLE_RAG=1

python3 tests/test_rag_ingest.py
python3 tests/test_suite.py   # regression

curl -X POST localhost:8000/v1/rag/corpora/ingest \
  -H 'Content-Type: application/json' \
  -d '{"corpus_id":"kb1","sources":[{"type":"local","path":"tests/fixtures/rag_corpus"}]}'

curl -s localhost:8000/v1/rag/corpora | jq .
```

### 2. Performance benchmarks

```bash
# Ingest throughput on fixture corpus (MB/s)
/usr/bin/time -f 'ingest_wall=%e maxrss=%M KB' \
  curl -X POST localhost:8000/v1/rag/corpora/ingest \
    -H 'Content-Type: application/json' \
    -d '{"corpus_id":"bench","sources":[{"type":"local","path":"tests/fixtures/rag_corpus"}]}'

python3 tests/test_rag_ingest.py -v
# Document ingest MB/s + index build time in documentation/reports/PHASE05_BENCH.md
# Pass: Blake3 dedup verified; index build completes within corpus MB limits
```

### 3. Memory leak & RSS audits

```bash
python3 tests/test_rag_ingest.py
python3 tests/memory_rss_audit.py
./scripts/valgrind.sh
# Re-ingest same corpus 5× — chunk store must not grow unbounded
for i in $(seq 1 5); do python3 tests/test_rag_ingest.py; done
# Pass: dedup keeps chunk count stable; RSS within NANOSERVE_CORPUS_* caps
```

### 4. Load & stress tests

```bash
export NANOSERVE_ENABLE_RAG=1
python3 tests/load_test_report.py --preset 50 --device cpu --out documentation/reports/PHASE05_LOAD.json
# Pass: inference unaffected; ingest API stable under background load

python3 tests/test_rag_ingest.py
```

### Post-build verification matrix

| Category | Command / artifact | Pass criteria |
|----------|-------------------|---------------|
| Unit / integration | `test_rag_ingest.py, test_suite.py` | Dedup + index PASS |
| Performance | `corpus ingest benchmark` | MB/s documented; within RAM caps |
| Memory leak / RSS | `re-ingest loop + memory_rss_audit.py` | No unbounded chunk growth |
| Load / stress | `load_test_report.py --preset 50` | ≥98% infer success with RAG ingest on |

**Sign-off:** Record results in `documentation/reports/PHASE05_VERIFY.md` (create if missing). CI must run sections 1–4 on every phase merge.

---

## Human checkpoint

| # | What you do | What you should see |
|---|-------------|---------------------|
| 1 | Ingest 1 MB local fixture corpus | `manifest.json` + `index.hnsw` under corpora dir |
| 2 | Ingest same content twice | Blake3 dedup — duplicate chunks stored once |
| 3 | `GET /v1/rag/corpora` | Lists `kb1` with metadata |
| 4 | Ingest from mock NMDP peer (test) | Staging shard → index entries |
| 5 | Run `tests/test_rag_ingest.py` internal search | Top-k chunk ids with scores (no public query API yet) |
| 6 | Inspect chunk files | Content-addressed `{blake3_hex}.chunk` blobs |

---

## Acceptance checklist

- [ ] **Post-build gate:** unit/integration + performance + memory leak/RSS + load/stress (see Automated verification); `PHASE05_VERIFY.md` recorded
- [ ] Ingest 1 MB fixture; index builds
- [ ] Internal index search returns top-k chunk ids with scores (`tests/test_rag_ingest.py`; public query API in Phase 06)
- [ ] Blake3 dedup verified on duplicate ingest
- [ ] Mesh ingest path works with mock peer agent
- [ ] `NANOSERVE_ENABLE_RAG=0` — RAG routes disabled; inference unchanged
- [ ] Default pip install size unchanged without `[rag]`

---

## Do not break

- `/v1/completions` without RAG flags
- GGUF and native inference paths
- NMDP security model from Phase 04

---

## Next phase

[TODO-Phase-06-RAG-Retrieval-Sessions-UI.md](TODO-Phase-06-RAG-Retrieval-Sessions-UI.md)
---

## Appendix — Phases R0–R1 (full spec)

> Verbatim from [TODO-NanoServe-Industry-grade-plan.md](../TODO-NanoServe-Industry-grade-plan.md)

## Phase R0 — Corpus + index specification

### On-disk layout

```
~/.nanoserve/corpora/{corpus_id}/
  manifest.json          # chunk catalog + metadata
  index.hnsw             # mmap vector index (Rust)
  index.meta.json        # dim, metric, quant dtype
  chunks/
    {blake3_hex}.chunk   # content-addressed, mmap-readable
```

### Corpus manifest entry

```json
{
  "chunk_id": "c_001",
  "hash": "blake3:...",
  "source": "peer:laptop-alice|local:./docs/foo.md",
  "offset": 0,
  "length": 4096,
  "mime": "text/markdown",
  "meta": { "title": "...", "page": 1 }
}
```

### Vector index (Rust: `rust/nanoq_rag/`)

- **Algorithm:** HNSW default; IVF-PQ optional for very large corpora
- **Quantization:** int8 embeddings default (384-dim → ~384 bytes/vector + graph overhead)
- **Metric:** cosine (normalize on ingest)
- **Embedding model:** small GGUF embedder (e.g. gte-small) via existing GGUF path, or `.nanoq` embed head post v3

### Files to add

| File | Action |
|------|--------|
| `rust/nanoq_rag/Cargo.toml` | New crate: HNSW, BM25, chunk store |
| `rust/nanoq_rag/src/index.rs` | mmap HNSW insert/search |
| `rust/nanoq_rag/src/chunk_store.rs` | Blake3-addressed blobs |
| `nanoserve/rag/spec.py` | Manifest schema, corpus id helpers |

**Acceptance:** Ingest 1 MB fixture corpus; index builds; query returns top-k chunk ids with scores.

---

## Phase R1 — Ingest pipeline

### Chunkers

| Format | Handler |
|--------|---------|
| Plain text / markdown | Fixed-size + paragraph-aware splits |
| JSONL | `{ "text": "..." }` or Q&A pairs |
| PDF | Optional `[rag]` extra (`pypdf` or similar) |

### Pipeline flow

```mermaid
flowchart LR
  Source[Local_or_NMDP] --> Chunk[Chunker]
  Chunk --> Hash[Blake3_dedup]
  Hash --> Embed[Embed_batch]
  Embed --> Index[HNSW_insert]
  Index --> Manifest[Update_manifest]
```

- Embed batching with backpressure; disk-spill queue for large ingest
- **Distributed ingest:** coordinator creates NMDP job → peers run data-agent → coordinator pulls shards → dedup by hash → local index

### API

```python
POST /v1/rag/corpora/ingest
{
  "corpus_id": "my-kb",
  "sources": [
    { "type": "local", "path": "/path/to/docs" },
    { "type": "mesh", "job_id": "uuid", "peer_device_ids": ["laptop-alice"] }
  ],
  "chunk_size": 512,
  "chunk_overlap": 64
}
```

**Acceptance:** Ingest from local path + mock peer agent; duplicate chunks stored once (Blake3 dedup).


### Phase 05 note on acceptance

Ingest phase validates **index build + internal search** via tests. Public `POST /v1/rag/corpora/{id}/query` ships in Phase 06 (R2).
