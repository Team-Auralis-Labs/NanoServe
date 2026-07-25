#!/bin/bash
set -euo pipefail
BUILD_DIR="/home/anand/AURALIS_LABS/LLMOrchestrator/NanoServe/engine/build"
LOG="/home/anand/AURALIS_LABS/LLMOrchestrator/NanoServe/build_full_output.txt"
exec > >(tee "$LOG") 2>&1

cd "$BUILD_DIR"
rm -rf ./*
cmake .. -DNANOSERVE_ENABLE_CUDA=ON -DNANOSERVE_ENABLE_OPENCL=OFF -DCMAKE_CUDA_ARCHITECTURES=75
make -j4
echo "BUILD_EXIT_CODE=0"
ls -la libnanoserve_engine.so
