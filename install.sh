#!/usr/bin/env bash
# NanoServe native installer (no Docker)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENABLE_CUDA="${ENABLE_CUDA:-0}"
ENABLE_OPENCL="${ENABLE_OPENCL:-0}"

echo "[*] NanoServe install — $ROOT"

if command -v apt-get >/dev/null 2>&1; then
  echo "[*] Installing system deps..."
  sudo apt-get update
  sudo apt-get install -y build-essential cmake curl python3 python3-pip python3-venv
  if [ "$ENABLE_OPENCL" = "1" ]; then
    sudo apt-get install -y ocl-icd-opencl-dev opencl-headers
  fi
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "[*] Installing Rust..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env"
fi

echo "[*] Building allocator..."
(cd allocator && cargo build --release)

echo "[*] Building engine (CUDA=$ENABLE_CUDA OPENCL=$ENABLE_OPENCL)..."
mkdir -p engine/build
(
  cd engine/build
  CMAKE_ARGS=()
  [ "$ENABLE_CUDA" = "1" ] && CMAKE_ARGS+=(-DNANOSERVE_ENABLE_CUDA=ON)
  [ "$ENABLE_OPENCL" = "1" ] && CMAKE_ARGS+=(-DNANOSERVE_ENABLE_OPENCL=ON)
  cmake .. "${CMAKE_ARGS[@]}"
  make -j"$(nproc)"
)

LIB="$ROOT/engine/build/libnanoserve_engine.so"
if [ ! -f "$LIB" ]; then
  echo "[!] Build failed: $LIB not found" >&2
  exit 1
fi

echo "[*] Installing Python package..."
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
pip install -q -e ".[server]"

ENV_FILE="$ROOT/.env.nanoserve"
cat > "$ENV_FILE" <<EOF
export LD_LIBRARY_PATH="$ROOT/allocator/target/release:\$LD_LIBRARY_PATH"
export NANOSERVE_ENGINE_LIB="$ROOT/engine/build/libnanoserve_engine.so"
export PYTHONPATH="$ROOT:\$PYTHONPATH"
export NANOSERVE_NUM_WORKERS="\$(nproc)"
export NANOSERVE_MAX_BATCH="32"
export NANOSERVE_MAX_QUEUE="512"
EOF

echo ""
echo "[+] Done. Run:"
echo "    source .venv/bin/activate && source .env.nanoserve"
echo "    ./scripts/run_native.sh           # dev / ~150 users"
echo "    ./scripts/run_native_300.sh       # production / 300 users"
echo ""
echo "Optional GPU build: ENABLE_CUDA=1 ./install.sh"
