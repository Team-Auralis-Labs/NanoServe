#!/usr/bin/env python3
"""Tests for .nanoq v3 archive loader."""
from __future__ import annotations

import ctypes
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LD_LIBRARY_PATH", ":".join([
    str(ROOT / "allocator" / "target" / "release"),
    os.environ.get("LD_LIBRARY_PATH", ""),
]))
os.environ.setdefault("NANOSERVE_ENGINE_LIB", str(ROOT / "engine" / "build" / "libnanoserve_engine.so"))

from nanoserve import Quantizer
from nanoserve.engine.worker import BackendKind, EngineWorker

FIXTURE = ROOT / "tests" / "fixtures" / "distilgpt2-int8.nanoq"


class TestNanoqV3Loader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib_path = os.environ["NANOSERVE_ENGINE_LIB"]
        if not Path(cls.lib_path).exists():
            raise unittest.SkipTest("Engine not built")
        if not FIXTURE.exists():
            raise unittest.SkipTest(f"Fixture missing: {FIXTURE}")
        cls.lib = ctypes.CDLL(cls.lib_path)
        cls.lib.nanoq_archive_validate_path.restype = ctypes.c_int
        cls.lib.nanoq_archive_validate_path.argtypes = [ctypes.c_char_p]

    def test_model_info_v3(self):
        worker = EngineWorker(lib_path=self.lib_path, backend=BackendKind.CPU, model_path=str(FIXTURE))
        meta = json.loads(worker.model_info())
        self.assertEqual(meta.get("format"), "nanoq_v3")
        self.assertEqual(meta.get("arch"), "gpt2")
        self.assertEqual(meta.get("n_layers"), 6)
        self.assertEqual(meta.get("vocab_size"), 50257)
        self.assertFalse(meta.get("legacy_demo", True))
        worker.cleanup()

    def test_v3_infer_coherent_english(self):
        worker = EngineWorker(lib_path=self.lib_path, backend=BackendKind.CPU, model_path=str(FIXTURE))
        out = worker.infer("Hello", 16)
        self.assertTrue(out)
        demo_words = {"the", "model", "is", "fast", "and", "efficient", "quantized", "inference"}
        words = set(out.lower().split())
        self.assertFalse(words.issubset(demo_words), f"Output looks like GEMV demo: {out!r}")
        self.assertIn("the", out.lower(), f"Expected coherent English continuation: {out!r}")
        worker.cleanup()

    def test_blake3_footer_rejects_tamper(self):
        data = bytearray(FIXTURE.read_bytes())
        data[-1] ^= 0xFF
        with tempfile.NamedTemporaryFile(suffix=".nanoq", delete=False) as f:
            f.write(data)
            bad = f.name
        try:
            rc = self.lib.nanoq_archive_validate_path(bad.encode())
            self.assertNotEqual(rc, 0)
            worker = EngineWorker(
                lib_path=self.lib_path, backend=BackendKind.CPU, model_path=bad
            )
            with self.assertRaises(RuntimeError):
                worker.model_info()
            worker.cleanup()
        finally:
            Path(bad).unlink(missing_ok=True)

    def test_legacy_v2_still_loads(self):
        path = Path(tempfile.gettempdir()) / "nanoserve_phase01_v2.nanoq"
        rng = np.random.default_rng(3)
        Quantizer.from_weights(rng.standard_normal((32, 64)).astype(np.float32), str(path), precision="int8")
        worker = EngineWorker(lib_path=self.lib_path, backend=BackendKind.CPU, model_path=str(path))
        meta = json.loads(worker.model_info())
        self.assertTrue(meta.get("legacy_demo"))
        self.assertTrue(worker.infer("legacy", 4))
        worker.cleanup()

    def test_engine_reset_kv_isolated_prompts(self):
        worker = EngineWorker(lib_path=self.lib_path, backend=BackendKind.CPU, model_path=str(FIXTURE))
        a = worker.infer("Hello", 8)
        worker.reset_kv()
        b = worker.infer("Hello", 8)
        self.assertEqual(a, b)
        worker.cleanup()

    def test_max_seq_bounds_no_crash(self):
        worker = EngineWorker(lib_path=self.lib_path, backend=BackendKind.CPU, model_path=str(FIXTURE))
        meta = json.loads(worker.model_info())
        # distilgpt2 n_positions=1024; engine must not exceed wpe rows
        self.assertLessEqual(meta.get("max_seq_len", 9999), 1024)
        for _ in range(3):
            out = worker.infer("Hello", 16)
            self.assertTrue(out)
        worker.cleanup()


if __name__ == "__main__":
    unittest.main()
