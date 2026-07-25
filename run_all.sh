#!/bin/bash
set -uo pipefail
ROOT="/home/anand/AURALIS_LABS/LLMOrchestrator/NanoServe"
OUT="$ROOT/full_run_output.txt"
exec > "$OUT" 2>&1

echo "=== BUILD ==="
cd "$ROOT/engine/build"
rm -rf ./*
chmod +x "$ROOT/engine/nvcc_wrapper.sh"
cmake .. -DNANOSERVE_ENABLE_CUDA=ON -DNANOSERVE_ENABLE_OPENCL=OFF -DCMAKE_CUDA_ARCHITECTURES=75
BUILD_EXIT=$?
echo "CMAKE_EXIT=$BUILD_EXIT"
if [ "$BUILD_EXIT" -ne 0 ]; then
  echo "BUILD_EXIT_CODE=$BUILD_EXIT"
  exit "$BUILD_EXIT"
fi

make -j4
BUILD_EXIT=$?
echo "BUILD_EXIT_CODE=$BUILD_EXIT"

echo "=== LIB CHECK ==="
if [ -f libnanoserve_engine.so ]; then
  ls -la libnanoserve_engine.so
  echo "LIB_PRODUCED=yes"
else
  echo "LIB_PRODUCED=no"
  ls -la . || true
fi

if [ "$BUILD_EXIT" -ne 0 ]; then
  exit "$BUILD_EXIT"
fi

echo "=== PYTHON PROBE ==="
export LD_LIBRARY_PATH="$ROOT/allocator/target/release"
export NANOSERVE_ENGINE_LIB="$ROOT/engine/build/libnanoserve_engine.so"
export PYTHONPATH="$ROOT"
python3 -c "
from nanoserve.engine.worker import EngineWorker, BackendKind
print('cuda probe', EngineWorker.probe_cuda())
cpu=EngineWorker(backend=BackendKind.CPU)
gpu=EngineWorker(backend=BackendKind.CUDA)
a=cpu.infer('parity test prompt', 16)
b=gpu.infer('parity test prompt', 16)
print('cpu', repr(a[:60]))
print('gpu', repr(b[:60]))
print('parity', a==b)
cpu.cleanup(); gpu.cleanup()
"

echo "=== TEST SUITE (last 25 lines) ==="
python3 "$ROOT/tests/test_suite.py" 2>&1 | tail -25
