# Quick test — tiny GGUF models

Try a **real small AI model** (not the fake demo words) on your laptop, then open it from your phone or another computer.

**Who this is for:** beginners. No jargon required.

---

## What is GGUF? (30 seconds)

Think of a **GGUF file** like a **compressed game save** for an AI brain.

- **Default Docker** (`docker compose up`) = demo mode, fake words
- **GGUF Docker** = real AI text, using a `.gguf` file you download

You need the **GGUF Docker profile** for this guide.

---

## Pick a small model (all on Hugging Face)

| Model | File to download | Size | Good for |
|-------|------------------|------|----------|
| **distilgpt2** | `distilgpt2-Q2_K.gguf` | ~61 MB | Smallest **chat** test (fits ~60 MB budget) |
| **SmolLM-135M** | `SmolLM-135M-Q2_K.gguf` | ~88 MB | Better tiny chat, a bit bigger |
| **gte-small** | `gte-small.Q2_K.gguf` | ~25 MB | **Not chat** — embeddings only, skip for “Generate” tests |

**Where to get them:**

- distilgpt2: https://huggingface.co/tensorblock/distilgpt2-GGUF  
- SmolLM-135M: https://huggingface.co/neopolita/smollm-135m-gguf  

Download **one** `.gguf` file into a folder called `models` in your NanoServe project:

```
NanoServe/
  models/
    distilgpt2-Q2_K.gguf    ← example
```

---

## Step 1 — Start the GGUF server (Docker)

Open a terminal in the NanoServe folder:

```bash
mkdir -p models
# put your .gguf file inside models/

export NANOSERVE_MODEL_PATH=/models/distilgpt2-Q2_K.gguf
docker compose --profile gguf up --build
```

Wait until it says the server is running. Open in your browser:

**http://localhost:8002**

(Port **8002** = GGUF Docker. Normal CPU Docker uses **8000**.)

**Check it worked:**

```bash
curl -s http://localhost:8002/health | jq '.gguf_available, .gguf_model_loaded'
```

You want `true` for both (after first generate, model loads).

### Native install (no Docker)

```bash
ENABLE_GGUF=1 ./install.sh
export NANOSERVE_MODEL_PATH=/path/to/distilgpt2-Q2_K.gguf
source .venv/bin/activate && source .env.nanoserve
./scripts/run_native.sh
```

Then use **http://localhost:8000** instead of 8002.

---

## Step 2 — Test in the Web UI (easiest)

1. Open **http://localhost:8002**
2. **Format** → pick **GGUF**
3. **Compute engine** → **CPU** (fine for tiny models)
4. Type a short prompt, e.g. `Hello, how are you?`
5. Click **Generate**

You should get **real AI text** (may be silly — these models are tiny).

If you see the old demo words (“the model is fast and efficient…”), you forgot **Format = GGUF**.

---

## Step 3 — Test in the TUI (terminal chat)

On the **same computer** as the server:

```bash
pip install httpx rich
python tui/client.py http://127.0.0.1:8002 --format gguf --device cpu
```

Type a message and press Enter. Try slash commands:

```
/format gguf
/device cpu
/help
```

---

## Step 4 — Test with code (SDK / API)

The Python **SDK** on your laptop talks to the **local engine file**, not Docker over the network. For Docker GGUF, use **HTTP** (same as the Web UI uses):

```python
import httpx

r = httpx.post(
    "http://127.0.0.1:8002/v1/completions",
    json={
        "prompt": "Once upon a time",
        "max_tokens": 40,
        "format": "gguf",
        "device": "cpu",
    },
    timeout=120.0,
)
print(r.json()["text"])
```

**One-line test in terminal:**

```bash
curl -X POST http://localhost:8002/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":32,"format":"gguf","device":"cpu"}'
```

---

## Use from another phone or laptop (same Wi‑Fi)

Your **host computer** runs Docker. Other devices talk to its **IP address**, not `localhost`.

### Find the host IP

On the computer running Docker:

```bash
hostname -I | awk '{print $1}'
```

Example: `192.168.1.42`

### From a phone or tablet — Web only

1. Phone must be on the **same Wi‑Fi** (not guest network)
2. Open browser: **http://192.168.1.42:8002**
3. Format = **GGUF** → Generate

If it does not load, open port **8002** on the host firewall (see [connect-network.md](connect-network.md)).

### From another laptop — Web or TUI

**Web:** same URL in Chrome/Firefox — `http://192.168.1.42:8002`

**TUI:**

```bash
python tui/client.py http://192.168.1.42:8002 --format gguf --device cpu
```

**API / Python from another laptop:**

```python
import httpx
r = httpx.post(
    "http://192.168.1.42:8002/v1/completions",
    json={"prompt": "Hi from another laptop", "max_tokens": 24, "format": "gguf"},
    timeout=120.0,
)
print(r.json()["text"])
```

Phones cannot run the TUI (needs Python in a terminal). Use the **browser** on phones.

---

## Quick comparison — which model when?

| You want… | Use this |
|-----------|----------|
| Smallest real chat test (~60 MB) | **distilgpt2-Q2_K.gguf** |
| Slightly better tiny chat (~88 MB) | **SmolLM-135M-Q2_K.gguf** |
| Just testing download size | **gte-small** — but it won’t chat |
| Default Docker, no download | Demo mode only — not real GGUF |

---

## Troubleshooting (simple)

| Problem | Fix |
|---------|-----|
| “GGUF off” in Web UI | Use port **8002** and `--profile gguf` |
| Still fake demo text | Set **Format → GGUF** |
| `gguf_model_loaded: false` | Check `NANOSERVE_MODEL_PATH` points to your file inside `/models/` |
| Phone can’t connect | Same Wi‑Fi; use host IP not `localhost`; open firewall port 8002 |
| Download failed in Web | GGUF path uses a **file you copied**, not HF download — put `.gguf` in `models/` folder |
| Very slow | Normal on CPU for first run — model loads once |

---

## Cheat sheet

```bash
# 1. Put model in ./models/
# 2. Start
export NANOSERVE_MODEL_PATH=/models/distilgpt2-Q2_K.gguf
docker compose --profile gguf up --build

# 3. This computer
#    Web → http://localhost:8002  (Format: GGUF)
#    TUI → python tui/client.py http://127.0.0.1:8002 --format gguf

# 4. Other devices (replace IP)
#    Web → http://192.168.1.42:8002
#    TUI → python tui/client.py http://192.168.1.42:8002 --format gguf
```

---

## See also

| Doc | What |
|-----|------|
| [connect-network.md](connect-network.md) | More on Wi‑Fi, firewall, mesh |
| [How-to-add-models-doc.md](How-to-add-models-doc.md) | All ways to add models |
| [Quick-deploy-method.md](Quick-deploy-method.md) | Docker CPU / GPU / GGUF ports |
