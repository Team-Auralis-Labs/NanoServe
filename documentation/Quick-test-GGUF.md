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

---

## How to download the model file (pick one way)

You need **one** `.gguf` file on your computer before starting Docker.  
These models are free on **Hugging Face** (like an app store for AI files).

### First — make the folder

Open a terminal in your NanoServe project folder and run:

```bash
cd /path/to/NanoServe
mkdir -p models
```

When done, it should look like:

```
NanoServe/
  models/
    distilgpt2-Q2_K.gguf    ← you will put the file here
```

---

### Way A — Download in the browser (easiest)

Good if you prefer clicking instead of typing commands.

**For distilgpt2 (~61 MB):**

1. Open: https://huggingface.co/tensorblock/distilgpt2-GGUF  
2. Click the **Files and versions** tab (or scroll to the file list).  
3. Find **`distilgpt2-Q2_K.gguf`** (~61 MB).  
4. Click the **↓ download** icon next to that file.  
5. Wait — it can take a few minutes on slow Wi‑Fi.  
6. Move the file from your **Downloads** folder into `NanoServe/models/`:

   ```bash
   mv ~/Downloads/distilgpt2-Q2_K.gguf ./models/
   ```

**For SmolLM-135M (~88 MB):**

1. Open: https://huggingface.co/neopolita/smollm-135m-gguf  
2. **Files and versions** → find **`SmolLM-135M-Q2_K.gguf`**.  
3. Download it.  
4. Move it:

   ```bash
   mv ~/Downloads/SmolLM-135M-Q2_K.gguf ./models/
   ```

**Check the file is really there:**

```bash
ls -lh models/
```

You should see your `.gguf` file and a size like `61M` or `88M`.

---

### Way B — Download in the terminal (one command)

Good if you already use the terminal. Needs **internet**.

**distilgpt2:**

```bash
mkdir -p models
curl -L -o models/distilgpt2-Q2_K.gguf \
  "https://huggingface.co/tensorblock/distilgpt2-GGUF/resolve/main/distilgpt2-Q2_K.gguf"
```

**SmolLM-135M:**

```bash
mkdir -p models
curl -L -o models/SmolLM-135M-Q2_K.gguf \
  "https://huggingface.co/neopolita/smollm-135m-gguf/resolve/main/SmolLM-135M-Q2_K.gguf"
```

`-L` follows redirects. `-o` saves with the right filename.

Verify:

```bash
ls -lh models/*.gguf
```

---

### Way C — Hugging Face CLI (optional)

Install once:

```bash
pip install huggingface-hub
```

Download distilgpt2:

```bash
mkdir -p models
huggingface-cli download tensorblock/distilgpt2-GGUF \
  distilgpt2-Q2_K.gguf \
  --local-dir models \
  --local-dir-use-symlinks False
```

The file ends up at `models/distilgpt2-Q2_K.gguf`.

---

### Match the filename to Docker

When you start the server, `NANOSERVE_MODEL_PATH` must use the **exact name** of the file you downloaded:

| If you downloaded… | Use this export |
|------------------|-----------------|
| `distilgpt2-Q2_K.gguf` | `export NANOSERVE_MODEL_PATH=/models/distilgpt2-Q2_K.gguf` |
| `SmolLM-135M-Q2_K.gguf` | `export NANOSERVE_MODEL_PATH=/models/SmolLM-135M-Q2_K.gguf` |

Docker maps your laptop folder `./models` → `/models` inside the container.  
So `/models/...` in the export = file inside your `NanoServe/models/` folder.

**Do not** use the Web UI “Download model” button for GGUF in this guide — that path is for other weight types. For GGUF, you download the `.gguf` file yourself (ways A–C above), then start Docker.

---

## Step 1 — Start the GGUF server (Docker)

Open a terminal in the NanoServe folder:

```bash
mkdir -p models
# (download .gguf into models/ first — see "How to download" above)

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

## Optional — GPU quick test (NVIDIA, native)

**Short version:** GGUF on **GPU** works best with a **native install** on your laptop (not the default GGUF Docker image). Docker port **8002** is **CPU-only** unless you customize the image yourself.

You need:

- An **NVIDIA** graphics card + driver installed  
- **CUDA** working on the host (`nvidia-smi` should print a table)

### 1. Install NanoServe with GGUF

```bash
ENABLE_GGUF=1 ./install.sh
source .venv/bin/activate && source .env.nanoserve
```

The default `pip install llama-cpp-python` is **CPU-only**. Reinstall it **with CUDA** so GGUF can use the GPU:

```bash
pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

*(This step can take several minutes — it compiles for your GPU.)*

### 2. Point at your model and turn on GPU layers

```bash
export NANOSERVE_MODEL_PATH="$PWD/models/distilgpt2-Q2_K.gguf"
export NANOSERVE_GGUF_N_GPU_LAYERS=99
```

- **`99`** = “offload as many layers as possible to the GPU” (fine for tiny models).  
- **`0`** = CPU only (Docker default).

### 3. Start the server

```bash
./scripts/run_native.sh
```

Open **http://localhost:8000**

### 4. Test GPU in the Web UI

1. **Format** → **GGUF**  
2. **Compute engine** → **GPU** (or **Auto**)  
3. Prompt → **Generate**

In the response meta you should see `device: gpu`. If you see a **warning** like “GPU requested but N_GPU_LAYERS=0”, check step 2.

**Health check:**

```bash
curl -s http://localhost:8000/health | jq '.gguf_available, .gpu_cuda, .gpu_available'
```

### 5. Test GPU in the TUI

```bash
python tui/client.py http://127.0.0.1:8000 --format gguf --device gpu
```

Or mid-chat: `/device gpu`

### 6. Test GPU with API / Python

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello GPU","max_tokens":32,"format":"gguf","device":"gpu"}'
```

Look for `"device": "gpu"` in the JSON.

```python
import httpx
r = httpx.post(
    "http://127.0.0.1:8000/v1/completions",
    json={"prompt": "Hello GPU", "max_tokens": 32, "format": "gguf", "device": "gpu"},
    timeout=120.0,
)
print(r.json()["device"], r.json()["text"])
```

### GPU from another laptop on Wi‑Fi

Same as CPU, but pick **GPU** in the Web UI or use `--device gpu` in the TUI:

```bash
python tui/client.py http://192.168.1.42:8000 --format gguf --device gpu
```

The **GPU runs on the host** that started `./scripts/run_native.sh` — your phone or other laptop only sends HTTP.

### Docker GGUF + GPU?

The stock **`docker compose --profile gguf`** image:

- Uses a **CPU** base image  
- Sets `NANOSERVE_GGUF_N_GPU_LAYERS=0`  
- Does **not** pass through NVIDIA GPUs  

So the quick-test Docker path stays **CPU**. For a GPU GGUF quick test, use **native** steps above.

| Path | GPU for GGUF? |
|------|----------------|
| Docker `--profile gguf` (port 8002) | **No** (CPU) — easiest quick test |
| Native + CUDA `llama-cpp-python` + `N_GPU_LAYERS=99` | **Yes** |

---

1. Open **http://localhost:8002** (Docker) or **http://localhost:8000** (native)
2. **Format** → pick **GGUF**
3. **Compute engine** → **CPU** (Docker / easiest) or **GPU** (native + [GPU setup](#optional--gpu-quick-test-nvidia-native) above)
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
| GGUF on GPU (NVIDIA) | Native install + CUDA `llama-cpp-python` — see [GPU section](#optional--gpu-quick-test-nvidia-native) |

---

## Troubleshooting (simple)

| Problem | Fix |
|---------|-----|
| “GGUF off” in Web UI | Use port **8002** and `--profile gguf` |
| Still fake demo text | Set **Format → GGUF** |
| `gguf_model_loaded: false` | Check `NANOSERVE_MODEL_PATH` points to your file inside `/models/` |
| Phone can’t connect | Same Wi‑Fi; use host IP not `localhost`; open firewall port 8002 |
| Download stuck or tiny file | Re-download; real file should be ~61 MB or ~88 MB, not a few KB |
| Wrong filename in Docker | `ls models/` and match `NANOSERVE_MODEL_PATH` exactly |
| Download failed in Web UI | Use browser/curl/CLI above — GGUF is not the Web “Download model” button |
| Very slow | Normal on CPU for first run — model loads once |
| Picked GPU but still CPU | Native only: reinstall `llama-cpp-python` with `CMAKE_ARGS=-DLLAMA_CUDA=on`; set `NANOSERVE_GGUF_N_GPU_LAYERS=99` |
| GPU warning in response | `N_GPU_LAYERS=0` — export `NANOSERVE_GGUF_N_GPU_LAYERS=99` and restart server |
| GPU on Docker 8002 | Stock GGUF Docker is CPU-only — use [native GPU section](#optional--gpu-quick-test-nvidia-native) |

---

## Cheat sheet

```bash
# 1. Download .gguf into ./models/  (browser, curl, or huggingface-cli — see above)
# 2. Start
export NANOSERVE_MODEL_PATH=/models/distilgpt2-Q2_K.gguf
docker compose --profile gguf up --build

# 3. This computer
#    Web → http://localhost:8002  (Format: GGUF)
#    TUI → python tui/client.py http://127.0.0.1:8002 --format gguf

# 4. Other devices (replace IP)
#    Web → http://192.168.1.42:8002
#    TUI → python tui/client.py http://192.168.1.42:8002 --format gguf

# --- Optional GPU (native, not Docker 8002) ---
# CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
# export NANOSERVE_GGUF_N_GPU_LAYERS=99
# export NANOSERVE_MODEL_PATH=$PWD/models/distilgpt2-Q2_K.gguf
# ./scripts/run_native.sh
# Web → Format GGUF, Compute engine GPU
# TUI → python tui/client.py http://127.0.0.1:8000 --format gguf --device gpu
```

---

## See also

| Doc | What |
|-----|------|
| [connect-network.md](connect-network.md) | More on Wi‑Fi, firewall, mesh |
| [How-to-add-models-doc.md](How-to-add-models-doc.md) | All ways to add models |
| [Quick-deploy-method.md](Quick-deploy-method.md) | Docker CPU / GPU / GGUF ports |
