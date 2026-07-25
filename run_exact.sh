#!/bin/bash
set -x
cd /home/anand/AURALIS_LABS/LLMOrchestrator/NanoServe/engine/build && rm -rf * && cmake .. -DNANOSERVE_ENABLE_CUDA=ON -DNANOSERVE_ENABLE_OPENCL=OFF && make -j4 2>&1
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"

ls -la /home/anand/AURALIS_LABS/LLMOrchestrator/NanoServe/engine/build/libnanoserve_engine.so 2>&1

if [ "$BUILD_EXIT" = "0" ]; then
export LD_LIBRARY_PATH=/home/anand/AURALIS_LABS/LLMOrchestrator/NanoServe/allocator/target/release
export NANOSERVE_ENGINE_LIB=/home/anand/AURALIS_LABS/LLMOrchestrator/NanoServe/engine/build/libnanoserve_engine.so
export PYTHONPATH=/home/anand/AURALIS_LABS/LLMOrchestrator/NanoServe
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
" 2>&1

python3 /home/anand/AURALIS_LABS/LLMOrchestrator/NanoServe/tests/test_suite.py 2>&1 | tail -25
fi
