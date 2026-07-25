# How to use NanoServe

After [SETUP.md](SETUP.md) or `./install.sh`, activate the environment:

```bash
source .venv/bin/activate && source .env.nanoserve
```

## Start the server

| Mode | Command | Users |
|------|---------|-------|
| Dev | `./scripts/run_native.sh` | ~50–150 |
| Production | `./scripts/run_native_300.sh` | ~300 |
| Docker CPU | `docker compose up --build` | demo |
| Docker GPU | `docker compose --profile gpu up --build` | demo + GPU |

Open **http://localhost:8000** for the Web UI.

## Web UI

1. Enter a prompt in the text box.
2. Choose **Compute Engine**: CPU, GPU, or Auto.
3. Click **Generate** — response shows tokens and which device ran.

## HTTP API

```bash
curl -s http://localhost:8000/health | jq .

curl -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain quantum dots in one sentence.","max_tokens":48,"device":"auto"}'
```

Fields:

- `prompt` — input text
- `max_tokens` — max generated tokens (default 64)
- `device` — `cpu`, `gpu`, or `auto` (GPU falls back to CPU with warnings)

## Python SDK

```bash
pip install -e .
python examples/sdk_demo.py
```

```python
from nanoserve import NanoServe

engine = NanoServe(device="auto")
text = engine.generate("Hello from the SDK", max_tokens=32)
print(text)
```

Async:

```python
import asyncio
from nanoserve import NanoServe

async def main():
    engine = NanoServe(device="cpu")
    text = await engine.generate_async("Async hello", max_tokens=24)
    print(text)

asyncio.run(main())
```

## Terminal UI (TUI)

```bash
pip install httpx rich
python tui/client.py --url http://localhost:8000 --device auto
```

Slash commands in the TUI:

- `/device cpu|gpu|auto` — switch backend
- `/quit` — exit

## Load testing

```bash
# 50 / 150 / 300 concurrent users (server must be running)
python3 tests/load_test_report.py --preset 50
python3 tests/load_test_report.py --preset 150
python3 tests/load_test_report.py --preset 300
```

## Device selection summary

| Client | How |
|--------|-----|
| Web UI | Dropdown |
| API | JSON `"device"` |
| SDK | `NanoServe(device="...")` |
| TUI | `--device` flag or `/device` |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `503 queue full` | Raise `NANOSERVE_MAX_QUEUE` or add gunicorn workers |
| Slow at 300 users | Use `./scripts/run_native_300.sh`, not single uvicorn |
| GPU warning, CPU used | Expected without CUDA/OpenCL build |
| `libnanoserve_engine.so` not found | `source .env.nanoserve` |

Reports: [reports/FULL_TEST_REPORT.md](reports/FULL_TEST_REPORT.md), [reports/STRESS_REPORT.md](reports/STRESS_REPORT.md), [reports/VALGRIND_REPORT.md](reports/VALGRIND_REPORT.md).
