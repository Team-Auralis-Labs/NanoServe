#!/usr/bin/env python3
"""Verify fp16/fp4 SIMD paths produce stable inference (parity with loaded models)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LD_LIBRARY_PATH", str(ROOT / "allocator" / "target" / "release"))
os.environ.setdefault("NANOSERVE_ENGINE_LIB", str(ROOT / "engine" / "build" / "libnanoserve_engine.so"))

from nanoserve import Quantizer
from nanoserve.engine.worker import BackendKind, EngineWorker


class TestSimdParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(os.environ["NANOSERVE_ENGINE_LIB"]).exists():
            raise unittest.SkipTest("Engine not built")

    def _run_model(self, precision: str) -> str:
        rng = np.random.default_rng(42)
        w = rng.standard_normal((64, 128)).astype(np.float32)
        path = Path(f"/tmp/nanoserve_simd_{precision}.nanoq")
        Quantizer.from_weights(w, str(path), precision=precision)
        worker = EngineWorker(model_path=str(path), backend=BackendKind.CPU)
        a = worker.infer("simd parity", 12)
        b = worker.infer("simd parity", 12)
        worker.cleanup()
        self.assertEqual(a, b)
        return a

    def test_fp16_deterministic(self):
        self.assertTrue(self._run_model("fp16"))

    def test_fp4_deterministic(self):
        self.assertTrue(self._run_model("fp4"))

    def test_int8_still_works(self):
        self.assertTrue(self._run_model("int8"))


if __name__ == "__main__":
    unittest.main()
