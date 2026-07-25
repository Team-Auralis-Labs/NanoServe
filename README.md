# NanoServe

<p align="center">
  <img src="documentation/assets/nanoserve-banner.png" alt="NanoServe — minimal LLM inference orchestrator" width="720"/>
</p>

<p align="center">
  <strong>Rust allocator · C++23 engine · Python SDK · FastAPI · Web UI · TUI</strong><br/>
  CPU (AVX2) + optional CUDA/OpenCL · Docker or native · scales to <strong>300 concurrent users</strong> without Docker
</p>

<p align="center">
  <a href="documentation/index.html">Documentation</a> ·
  <a href="documentation/SETUP.md">Setup</a> ·
  <a href="documentation/USAGE.md">Usage</a> ·
  <a href="documentation/reports/FULL_TEST_REPORT.md">Test report</a>
</p>

---

## Architecture

```mermaid
flowchart TB
  subgraph clients [Clients]
    SDK["Python SDK"]
    Web["Web UI"]
    TUI["TUI"]
  end
  LB["nginx :8000\n(300-user prod)"]
  API["FastAPI micro-batcher"]
  Pool["EnginePool"]
  CPU["CPUSimdBackend AVX2"]
  GPU["CUDA / OpenCL"]
  clients --> LB
  LB --> API
  SDK --> Pool
  API --> Pool
  Pool --> CPU
  Pool --> GPU
```

### Request flow (animated sequence)

```mermaid
sequenceDiagram
  participant U as User / Client
  participant F as FastAPI
  participant B as Micro-batcher
  participant E as EnginePool
  participant C as CPU/GPU

  U->>F: POST /v1/completions
  F->>B: enqueue prompt
  B->>B: collect batch (≤32, 25ms window)
  B->>E: infer batch
  E->>C: int8 GEMV
  C-->>E: tokens
  E-->>F: completions
  F-->>U: JSON + device + warnings
```

---

## Quick start

### Docker (fastest demo)

```bash
docker compose up --build              # CPU → http://localhost:8000
docker compose --profile gpu up --build   # GPU → http://localhost:8001
```

### Native (no Docker)

**Prior requirements:** [documentation/REQUIREMENTS.md](documentation/REQUIREMENTS.md) — OS packages, Rust, CMake, Python 3.10+, nginx for 300-user production.

```bash
git clone https://github.com/<your-org>/NanoServe.git && cd NanoServe
./install.sh                           # CPU build + venv + .env.nanoserve
ENABLE_CUDA=1 ./install.sh             # optional NVIDIA CUDA

source .venv/bin/activate && source .env.nanoserve
./scripts/run_native.sh                # dev / ~150 users
./scripts/run_native_300.sh            # production / 300 users (nginx + gunicorn)
```

Open **http://localhost:8000** — Web UI with **Compute Engine** dropdown (CPU / GPU / Auto).

---

## How to use

| Task | Command / link |
|------|----------------|
| Full usage guide | [documentation/USAGE.md](documentation/USAGE.md) |
| Web UI | http://localhost:8000 |
| Health check | `curl -s localhost:8000/health \| jq .` |
| Completions API | `POST /v1/completions` with `"device": "cpu\|gpu\|auto"` |
| Python SDK | `python examples/sdk_demo.py` |
| Terminal chat | `python tui/client.py --device auto` |
| Load test | `python3 tests/load_test_report.py --preset 300` |

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":24,"device":"auto"}'
```

```python
from nanoserve import NanoServe

engine = NanoServe(device="auto")
print(engine.generate("Hello world", max_tokens=50))
```

---

## Scaling (non-Docker)

| Tier | Users | Command |
|------|-------|---------|
| Dev | ≤50 | `./scripts/run_native.sh` |
| Medium | ~150 | `./scripts/run_native.sh` |
| **Production** | **300** | **`./scripts/run_native_300.sh`** |

Details: [documentation/SCALING.md](documentation/SCALING.md)

---

## Test & quality reports

Human-readable reports in **Markdown, HTML, and PDF**:

| Report | MD | HTML | PDF |
|--------|----|------|-----|
| Full test suite (14/14 pass) | [FULL_TEST_REPORT.md](documentation/reports/FULL_TEST_REPORT.md) | [HTML](documentation/reports/FULL_TEST_REPORT.html) | [PDF](documentation/reports/FULL_TEST_REPORT.pdf) |
| User stress (50/150/300) | [STRESS_REPORT.md](documentation/reports/STRESS_REPORT.md) | [HTML](documentation/reports/STRESS_REPORT.html) | [PDF](documentation/reports/STRESS_REPORT.pdf) |
| Valgrind (0 leaks) | [VALGRIND_REPORT.md](documentation/reports/VALGRIND_REPORT.md) | [HTML](documentation/reports/VALGRIND_REPORT.html) | [PDF](documentation/reports/VALGRIND_REPORT.pdf) |

Regenerate all docs:

```bash
pip install fpdf2
python3 scripts/generate_reports.py
```

Browse: [documentation/index.html](documentation/index.html)

---

## Naming

| Context | Name |
|---------|------|
| GitHub / marketing | **NanoServe** |
| Repo folder | `NanoServe` |
| pip install | `pip install nanoserve` |
| Python import | `from nanoserve import NanoServe` |
| C++ shared lib | `libnanoserve_engine.so` |

---

## Project layout

```
allocator/       Rust buddy-tier memory allocator
engine/          C++23 inference (CPU AVX2, CUDA, OpenCL)
nanoserve/       Python SDK (NanoServe, Quantizer, EnginePool)
server/          FastAPI + static Web UI
tui/             Terminal client + load tester
scripts/         install, production serve, reports, valgrind
documentation/   Setup, usage, requirements, reports (MD/HTML/PDF)
deployment/      nginx config for 300-user native tier
```

---

## Build options

| Flag | Default | Description |
|------|---------|-------------|
| `NANOSERVE_ENABLE_CUDA` | OFF | CUDA INT8 GEMV backend |
| `NANOSERVE_ENABLE_OPENCL` | OFF | OpenCL backend |

```bash
cd engine/build && cmake .. -DNANOSERVE_ENABLE_CUDA=ON && make -j$(nproc)
```

---

## Design notes

- **Backward-compatible FFI:** `engine_init()`, `engine_infer()`, `engine_cleanup()` unchanged on CPU path.
- **Device routing:** `device=cpu|gpu|auto` on API, SDK, Web UI, and TUI.
- **GPU fallback:** Graceful CPU fallback with warnings when GPU unavailable.
- **300-user native:** Same stack as Docker demo — nginx + 4× gunicorn via `run_native_300.sh`; not a separate codebase.

---

## License

See repository license file.
