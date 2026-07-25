#!/usr/bin/env python3
"""Valgrind harness: repeated engine init/infer/cleanup cycles."""
from nanoserve.engine.worker import BackendKind, EngineWorker
import os

lib = os.environ.get("NANOSERVE_ENGINE_LIB")
cycles = int(os.environ.get("VALGRIND_CYCLES", "100"))

for i in range(cycles):
    w = EngineWorker(lib_path=lib, backend=BackendKind.CPU)
    w.infer(f"valgrind cycle {i}", max_tokens=8)
    w.cleanup()

print(f"OK {cycles} cycles")
