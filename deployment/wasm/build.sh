#!/usr/bin/env bash
# Build NanoServe CPU engine → deployment/wasm/nanoserve_engine.{js,wasm}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/deployment/wasm"
BUILD="$ROOT/engine/build-wasm"

if ! command -v emcc >/dev/null 2>&1; then
  echo "[!] Emscripten (emcc) not found." >&2
  echo "    Install: https://emscripten.org/docs/getting_started/downloads.html" >&2
  echo "    Then:   source /path/to/emsdk/emsdk_env.sh" >&2
  exit 1
fi

echo "[*] Building WASM engine → $OUT"
mkdir -p "$OUT" "$OUT/assets"
rm -rf "$BUILD"
mkdir -p "$BUILD"

em++ -O3 -std=c++23 \
  -I"$ROOT/engine/include" \
  -DNANOSERVE_WASM=1 \
  "$ROOT/engine/src/engine_ffi.cpp" \
  "$ROOT/engine/src/engine_core.cpp" \
  "$ROOT/engine/src/nanoq_loader.cpp" \
  "$ROOT/engine/src/backend_cpu.cpp" \
  "$ROOT/engine/src/backend_factory.cpp" \
  "$ROOT/engine/src/backend_cuda_stub.cpp" \
  "$ROOT/engine/src/backend_opencl_stub.cpp" \
  "$ROOT/engine/src/buddy_pool_wasm.cpp" \
  -o "$OUT/nanoserve_engine.js" \
  -sWASM=1 \
  -sMODULARIZE=1 \
  -sEXPORT_NAME=createNanoServeModule \
  -sALLOW_MEMORY_GROWTH=1 \
  -sENVIRONMENT=web \
  -sEXPORTED_FUNCTIONS='["_engine_init","_engine_init_with_model_bytes","_engine_reload_model_bytes","_engine_infer","_engine_model_info","_engine_cleanup","_malloc","_free"]' \
  -sEXPORTED_RUNTIME_METHODS='["cwrap","UTF8ToString","getValue","HEAPU8"]' \
  -sFILESYSTEM=0 \
  -sNO_EXIT_RUNTIME=1

echo "[+] WASM artifacts:"
ls -lh "$OUT/nanoserve_engine.js" "$OUT/nanoserve_engine.wasm"

# Optional tiny demo .nanoq
if python3 -c "import nanoserve" 2>/dev/null; then
  python3 - <<PY
import sys
from pathlib import Path
import numpy as np
ROOT = Path("${ROOT}")
sys.path.insert(0, str(ROOT))
from nanoserve import Quantizer
out = ROOT / "deployment/wasm/assets/demo.nanoq"
w = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)
Quantizer.from_weights(w, str(out), precision="int8", name="wasm-demo")
print(f"[+] Demo model → {out}")
PY
else
  echo "[*] Skip demo model (nanoserve not installed in python)"
fi

echo "[+] Done. Serve: npx serve $OUT"
