#!/bin/bash
# Resolve nvcc across apt (/usr/lib/cuda) and NVIDIA container (/usr/local/cuda) layouts.
set -euo pipefail
NVCC=""
for candidate in \
  "${CUDA_HOME:-}/bin/nvcc" \
  "/usr/local/cuda/bin/nvcc" \
  "/usr/lib/cuda/bin/nvcc"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    NVCC="$candidate"
    break
  fi
done
if [ -z "$NVCC" ]; then
  echo "nvcc not found (set CUDA_HOME or install CUDA toolkit)" >&2
  exit 127
fi
CCBIN="${CMAKE_CUDA_HOST_COMPILER:-${CXX:-g++}}"
exec "$NVCC" -allow-unsupported-compiler -ccbin "$CCBIN" "$@"
