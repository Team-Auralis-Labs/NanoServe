#!/usr/bin/env python3
"""RSS cap during repeated native v3 inference."""
from __future__ import annotations

import json
import os
import resource
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LD_LIBRARY_PATH", ":".join([
    str(ROOT / "allocator" / "target" / "release"),
    str(ROOT / "rust" / "nanoq_runtime" / "target" / "release"),
    os.environ.get("LD_LIBRARY_PATH", ""),
]))
os.environ.setdefault("NANOSERVE_ENGINE_LIB", str(ROOT / "engine" / "build" / "libnanoserve_engine.so"))

from nanoserve.engine.worker import BackendKind, EngineWorker

FIXTURE = ROOT / "tests" / "fixtures" / "distilgpt2-int8.nanoq"


class TestNanoqMemory(unittest.TestCase):
    def test_rss_plateau(self):
        if not FIXTURE.exists():
            self.skipTest("Fixture missing")
        lib = os.environ.get("NANOSERVE_ENGINE_LIB", "")
        if not Path(lib).exists():
            self.skipTest("Engine not built")

        worker = EngineWorker(lib_path=lib, backend=BackendKind.CPU, model_path=str(FIXTURE))
        rss_samples = []
        for i in range(20):
            worker.infer(f"memory test prompt {i}", 8)
            rss_samples.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        worker.cleanup()
        # Allow modest growth; fail on runaway (>50% over min)
        lo = min(rss_samples)
        hi = max(rss_samples)
        self.assertLess(hi, lo * 1.5 + 1_000_000, f"RSS drift: {lo} -> {hi}")


if __name__ == "__main__":
    unittest.main()
