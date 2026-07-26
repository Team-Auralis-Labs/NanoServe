#!/usr/bin/env python3
"""Tests for .nanoq v2 format and C++ loader via FFI."""
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


class TestNanoqLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = os.environ["NANOSERVE_ENGINE_LIB"]
        if not Path(cls.lib).exists():
            raise unittest.SkipTest("Engine not built")

    def _write(self, path: Path, precision: str) -> None:
        rng = np.random.default_rng(7)
        w = rng.standard_normal((64, 128)).astype(np.float32)
        Quantizer.from_weights(w, str(path), precision=precision, name="test")

    def test_int8_roundtrip_and_load(self):
        path = Path("/tmp/nanoserve_nanoq_int8.nanoq")
        self._write(path, "int8")
        info = Quantizer.read_nanoq(str(path))
        self.assertEqual(info["header"]["dtype"], "int8")
        worker = EngineWorker(lib_path=self.lib, backend=BackendKind.CPU, model_path=str(path))
        out = worker.infer("nanoq int8", 8)
        self.assertTrue(out)
        meta = worker.model_info()
        self.assertIn("int8", meta)
        worker.cleanup()

    def test_fp16_load(self):
        path = Path("/tmp/nanoserve_nanoq_fp16.nanoq")
        self._write(path, "fp16")
        worker = EngineWorker(lib_path=self.lib, backend=BackendKind.CPU, model_path=str(path))
        a = worker.infer("fp16 test", 8)
        b = worker.infer("fp16 test", 8)
        self.assertEqual(a, b)
        worker.cleanup()

    def test_fp4_load(self):
        path = Path("/tmp/nanoserve_nanoq_fp4.nanoq")
        self._write(path, "fp4")
        worker = EngineWorker(lib_path=self.lib, backend=BackendKind.CPU, model_path=str(path))
        self.assertTrue(worker.infer("fp4 test", 8))
        worker.cleanup()

    def test_reload_model(self):
        p1 = Path("/tmp/nanoserve_nanoq_a.nanoq")
        p2 = Path("/tmp/nanoserve_nanoq_b.nanoq")
        self._write(p1, "int8")
        self._write(p2, "fp16")
        worker = EngineWorker(lib_path=self.lib, backend=BackendKind.CPU, model_path=str(p1))
        out1 = worker.infer("reload", 8)
        worker.reload_model(str(p2))
        out2 = worker.infer("reload", 8)
        worker.cleanup()
        self.assertTrue(out1)
        self.assertTrue(out2)


if __name__ == "__main__":
    unittest.main()
