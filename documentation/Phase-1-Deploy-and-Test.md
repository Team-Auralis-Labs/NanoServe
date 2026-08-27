# Phase 1 — Deploy & Test (Native `.nanoq` v3)

**What changed:** `format=nanoq` now runs a **real GPT-2 transformer** (distilgpt2-class), not the old 22-word GEMV demo. No llama-cpp required on this path.

---

## Prerequisites

- Linux, Python 3.10+, Rust, CMake, g++
- Repo cloned; run from project root

```bash
./install.sh          # or: pip install -e . && build steps below
```

---

## 1. Build

```bash
cd allocator && cargo build --release && cd ..
cd rust/nanoq_runtime && cargo build --release && cd ../..
cd engine/build && cmake .. -DNANOSERVE_LLAMA_CPP_KERNELS=1 && make -j$(nproc) && cd ../..
```

Set runtime paths (add to your shell or `.env.nanoserve`):

```bash
export LD_LIBRARY_PATH="$PWD/allocator/target/release:$LD_LIBRARY_PATH"
export NANOSERVE_ENGINE_LIB="$PWD/engine/build/libnanoserve_engine.so"
export PYTHONPATH="$PWD:$PYTHONPATH"
```

---

## 2. Get a v3 model (required for demo)

`*.nanoq` files are **gitignored**. Build the distilgpt2 fixture locally (~486 MB, ~10 min first time):

```bash
pip install blake3 transformers torch
python3 tests/fixtures/build_distilgpt2_v3_fixture.py
# → tests/fixtures/distilgpt2-int8.nanoq
```

---

## 3. Start server

```bash
source .venv/bin/activate 2>/dev/null || true
python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** or use curl below.

---

## 4. Test the new feature

### API (recommended)

```bash
FIXTURE="$PWD/tests/fixtures/distilgpt2-int8.nanoq"

curl -s http://localhost:8000/health | python3 -m json.tool

curl -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"Hello\",\"max_tokens\":32,\"format\":\"nanoq\",\"model\":\"$FIXTURE\",\"device\":\"cpu\"}"
```

**Success:** JSON with `"format":"nanoq"` and real English (e.g. *" The following is a new feature…"*).  
**Failure:** Repeated demo words (*"the model is fast and efficient…"*) → v3 model not loaded; check `model` path.

### Python worker (no server)

```bash
python3 -c "
import os, json
from nanoserve.engine.worker import BackendKind, EngineWorker
w = EngineWorker(
    lib_path=os.environ['NANOSERVE_ENGINE_LIB'],
    backend=BackendKind.CPU,
    model_path='tests/fixtures/distilgpt2-int8.nanoq',
)
print(json.loads(w.model_info()))   # arch=gpt2, n_layers=6, format=nanoq_v3
print(w.infer('Hello', 16))
w.cleanup()
"
```

### Automated gate (CI / local)

```bash
python3 tests/test_nanoq_v3_loader.py
python3 tests/test_tokenizer_rust.py
./engine/build/test_transformer_gpt2 tests/fixtures/distilgpt2-int8.nanoq
python3 tests/test_suite.py
```

Full audit details: [reports/Phase-1-Test-Report.md](reports/Phase-1-Test-Report.md)

---

## 5. What to verify

| Check | Expected |
|-------|----------|
| Model info | `"format":"nanoq_v3"`, `"arch":"gpt2"`, `"n_layers":6`, `"max_seq_len":1024` |
| Inference | Coherent English; **not** the 22-word synthetic vocab |
| Legacy v2 | Small single-matrix `.nanoq` still loads with `"legacy_demo":true` |
| Blake3 tamper | Corrupted footer rejected on load |
| GGUF path | Unchanged — see [Quick-test-GGUF.md](Quick-test-GGUF.md) |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Engine not built` | Run build steps; check `NANOSERVE_ENGINE_LIB` |
| Fixture missing | Run `build_distilgpt2_v3_fixture.py` |
| Still demo text | Pass full path to v3 `.nanoq` in `"model"` |
| `ImportError: blake3` | `pip install blake3 transformers torch` |
| Slow first request | ~2–3s cold; ~1s warm (CPU, distilgpt2) |
| High RAM (~1 GB) | Normal for fp32 v3 fixture + KV cache |

---

## See also

| Doc | Purpose |
|-----|---------|
| [reports/Phase-1-Test-Report.md](reports/Phase-1-Test-Report.md) | Full stress/memory/perf audit |
| [reports/PHASE01_VERIFY.md](reports/PHASE01_VERIFY.md) | Sign-off checklist |
| [SETUP.md](SETUP.md) | General install & Docker |
| [Quick-test-GGUF.md](Quick-test-GGUF.md) | Optional GGUF path (port 8002) |
