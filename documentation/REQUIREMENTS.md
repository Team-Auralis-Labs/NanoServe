# Non-Docker deployment — prior requirements

Install these **before** running `./install.sh`. The installer can install some packages on Debian/Ubuntu via `apt-get`; other distros need manual setup.

## Required (all deployments)

| Requirement | Minimum | Verify |
|-------------|---------|--------|
| **OS** | Linux x86_64 (Pop!\_OS, Ubuntu 20.04+, Debian 11+) | `uname -m` → `x86_64` |
| **RAM** | 8 GB (16 GB for 150+ users, 32 GB for 300 users) | `free -h` |
| **CPU** | 4+ cores (8+ for 300-user production) | `nproc` |
| **Disk** | ~2 GB free (build + venv) | `df -h .` |
| **build-essential** | gcc/g++, make | `gcc --version` |
| **cmake** | 3.16+ | `cmake --version` |
| **python3** | 3.10+ | `python3 --version` |
| **python3-venv, pip** | stdlib venv | `python3 -m venv --help` |
| **curl** | for Rust bootstrap | `curl --version` |
| **Rust (cargo)** | 1.70+ | `cargo --version` (installer adds if missing) |

Debian/Ubuntu one-liner:

```bash
sudo apt-get update && sudo apt-get install -y \
  build-essential cmake curl python3 python3-pip python3-venv git
```

## Required for 150–300 user production (non-Docker)

| Requirement | Purpose |
|-------------|---------|
| **gunicorn** | Multi-process HTTP (`pip install gunicorn`, added by `serve_production.sh`) |
| **nginx** | Load balancer on `:8000` → gunicorn workers on `:8001–8004` |
| **ulimit** | `nofile` ≥ 65535 recommended (`ulimit -n`) |

```bash
sudo apt-get install -y nginx
```

Use `./scripts/run_native_300.sh` after install — it sets worker counts and starts nginx + gunicorn.

## Optional — GPU backends

| Backend | Requirements |
|---------|--------------|
| **CUDA** | NVIDIA driver, CUDA 11.x toolkit, `g++-9` (see `engine/nvcc_wrapper.sh`) |
| **OpenCL** | `ocl-icd-opencl-dev`, `opencl-headers`, GPU ICD |

```bash
ENABLE_CUDA=1 ./install.sh
ENABLE_OPENCL=1 ./install.sh
```

## Optional — testing & profiling

| Tool | Purpose |
|------|---------|
| **valgrind** | C engine memory audit (`./scripts/valgrind.sh`) |
| **httpx, rich** | TUI client (`pip install httpx rich`) |

```bash
sudo apt-get install -y valgrind
```

## Scaling tiers (non-Docker)

| Tier | Script | nginx | Engine workers |
|------|--------|-------|----------------|
| Dev / demo | `./scripts/run_native.sh` | No | `nproc` (single process) |
| ~150 users | `./scripts/run_native.sh` + tune env | Optional | `NANOSERVE_NUM_WORKERS=$(nproc)` |
| **300 users** | **`./scripts/run_native_300.sh`** | **Yes** | **`nproc / 4` per gunicorn process × 4 processes** |

Environment (written by `install.sh` to `.env.nanoserve`):

```bash
export NANOSERVE_NUM_WORKERS="$(nproc)"      # overridden by run_native_300.sh
export NANOSERVE_MAX_BATCH="32"
export NANOSERVE_MAX_QUEUE="512"
export LD_LIBRARY_PATH="$PWD/allocator/target/release:$LD_LIBRARY_PATH"
export NANOSERVE_ENGINE_LIB="$PWD/engine/build/libnanoserve_engine.so"
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Network

- Port **8000** — HTTP API + Web UI (nginx in production mode)
- Ports **8001–8004** — internal gunicorn (production only)

## What the installer does not provide

- Docker / container runtime
- NVIDIA Container Toolkit (Docker GPU only)
- Systemd unit files (add your own for production)
- TLS certificates (terminate TLS at nginx or a reverse proxy)

See [SETUP.md](SETUP.md) and [USAGE.md](USAGE.md) for step-by-step setup and daily use.
