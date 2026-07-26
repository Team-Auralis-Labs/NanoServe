#!/usr/bin/env python3
"""GGUF probe, routing, and health tests (no llama-cpp required for most)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LD_LIBRARY_PATH", str(ROOT / "allocator" / "target" / "release"))
os.environ.setdefault("NANOSERVE_ENGINE_LIB", str(ROOT / "engine" / "build" / "libnanoserve_engine.so"))

from nanoserve.engine.gguf_probe import gguf_available
from nanoserve.engine.gguf_worker import active_format, default_gguf_path, gguf_model_loaded
from nanoserve.engine.router import InferenceRouter, resolve_format


class TestGGUF(unittest.TestCase):
    def test_probe_returns_bool(self):
        self.assertIsInstance(gguf_available(), bool)

    def test_resolve_gguf_path(self):
        self.assertEqual(resolve_format("/tmp/model.gguf", "auto"), "gguf")

    def test_forced_nanoq(self):
        self.assertEqual(resolve_format("/tmp/model.gguf", "nanoq"), "nanoq")

    def test_forced_gguf_without_path(self):
        self.assertEqual(resolve_format(None, "gguf"), "gguf")

    def test_active_format_default(self):
        self.assertEqual(active_format(), "nanoq")
        self.assertFalse(gguf_model_loaded())

    def test_default_gguf_path_env(self):
        with patch.dict(os.environ, {"NANOSERVE_MODEL_PATH": "/tmp/x.gguf"}):
            with patch("nanoserve.engine.gguf_worker.os.path.isfile", return_value=True):
                self.assertEqual(default_gguf_path(), "/tmp/x.gguf")

    def test_gguf_fallback_without_extra(self):
        if not Path(os.environ["NANOSERVE_ENGINE_LIB"]).exists():
            self.skipTest("Engine not built")
        router = InferenceRouter(num_workers=1)
        gguf = Path("/tmp/fake_gguf_test.gguf")
        gguf.write_bytes(b"GGUF")
        try:
            r = router.submit("hi", 4, model=str(gguf), format="gguf").result()
            if not router.gguf_available:
                self.assertEqual(r.format, "nanoq")
                self.assertTrue(any("GGUF" in w for w in r.warnings))
        finally:
            gguf.unlink(missing_ok=True)

    def test_format_nanoq_unchanged(self):
        if not Path(os.environ["NANOSERVE_ENGINE_LIB"]).exists():
            self.skipTest("Engine not built")
        router = InferenceRouter(num_workers=1)
        r = router.submit("hello", 8, format="nanoq").result()
        self.assertEqual(r.format, "nanoq")
        self.assertTrue(r.text)


if __name__ == "__main__":
    unittest.main()
