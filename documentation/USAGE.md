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
3. Select **Model** (**Built-in demo** on CPU `:8000`, or **Select model…** on GGUF `:8002`), **Format** (Auto / Native / GGUF), and **Precision** (int8 default, fp16, fp4, raw).
4. Use **Download model** for HuggingFace repo or direct URL.
5. Click **Generate** — response shows tokens, model, format, and device.

## HTTP API

```bash
curl -s http://localhost:8000/health | jq .

curl -s http://localhost:8000/v1/models | jq .

curl -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain quantum dots in one sentence.","max_tokens":48,"device":"auto","model":"my-model","format":"auto","precision":"int8"}'

curl -X POST http://localhost:8000/v1/models/download \
  -H 'Content-Type: application/json' \
  -d '{"source":"hf","repo_id":"org/model","filename":"model.safetensors","precision":"int8"}'
```

Fields:

- `prompt` — input text
- `max_tokens` — max generated tokens (default 64)
- `device` — `cpu`, `gpu`, or `auto` (GPU falls back to CPU with warnings)
- `model` — model id or path (optional)
- `format` — `auto`, `nanoq`, or `gguf`
- `quantize` — `true`/`false`/`null` (null uses `NANOSERVE_AUTO_QUANTIZE`, default on)
- `precision` — `int8` (default), `fp16`, `fp4`, or `raw` (unquantized, not recommended)

## Python SDK

```bash
pip install -e .
python examples/sdk_demo.py
```

```python
from nanoserve import NanoServe

engine = NanoServe(device="auto", model="my-model")
text = engine.generate("Hello from the SDK", max_tokens=32)
print(text)

# Download and register a model (requires pip install nanoserve[models])
engine.download("hf", repo_id="org/model", filename="model.safetensors")
print(engine.list_models())
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
- `/model [id|path]` — select model
- `/format auto|nanoq|gguf` — runtime format
- `/precision int8|fp16|fp4|raw` — quantization precision
- `/models` — list registered models
- `/download hf <repo> [file]` or `/download url <url>` — download model
- `/quit` — exit

## Load testing

```bash
# 50 / 150 / 300 concurrent users (server must be running)
python3 tests/load_test_report.py --preset 50
python3 tests/load_test_report.py --preset 150
python3 tests/load_test_report.py --preset 300
```

## Device selection summary

| Client | Device | Model / format / precision |
|--------|--------|----------------------------|
| Web UI | Compute Engine dropdown | Model, Format, Precision dropdowns + Download |
| API | JSON `"device"` | `"model"`, `"format"`, `"precision"`, `"quantize"` |
| SDK | `NanoServe(device=..., model=...)` | `generate(..., format=, precision=, quantize=)` |
| TUI | `--device` or `/device` | `/model`, `/format`, `/precision`, `/download` |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `503 queue full` | Raise `NANOSERVE_MAX_QUEUE` or add gunicorn workers |
| Slow at 300 users | Use `./scripts/run_native_300.sh`, not single uvicorn |
| GPU warning, CPU used | Expected without CUDA/OpenCL build |
| `libnanoserve_engine.so` not found | `source .env.nanoserve` |

Reports: [reports/FULL_TEST_REPORT.md](reports/FULL_TEST_REPORT.md), [reports/STRESS_REPORT.md](reports/STRESS_REPORT.md), [reports/VALGRIND_REPORT.md](reports/VALGRIND_REPORT.md).
