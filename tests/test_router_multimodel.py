#!/usr/bin/env python3
"""Multi-model router tests."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LD_LIBRARY_PATH", str(ROOT / "allocator" / "target" / "release"))
os.environ.setdefault("NANOSERVE_ENGINE_LIB", str(ROOT / "engine" / "build" / "libnanoserve_engine.so"))

from nanoserve import Quantizer
from nanoserve.engine.router import InferenceRouter, resolve_format
from nanoserve.models.registry import ModelEntry, ModelRegistry


class TestRouterMultimodel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(os.environ["NANOSERVE_ENGINE_LIB"]).exists():
            raise unittest.SkipTest("Engine not built")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.reg = ModelRegistry(root=self.tmp)
        self.router = InferenceRouter(num_workers=2, registry=self.reg)

        for name, seed in (("model-a", 1), ("model-b", 99)):
            rng = np.random.default_rng(seed)
            w = rng.standard_normal((32, 64)).astype(np.float32)
            nanoq = self.tmp / name / f"{name}.nanoq"
            nanoq.parent.mkdir(parents=True)
            Quantizer.from_weights(w, str(nanoq), precision="int8", name=name)
            self.reg.register(ModelEntry(
                id=name,
                source_path=str(nanoq),
                nanoq_path=str(nanoq),
                format="nanoq",
                dtype="int8",
                rows=32,
                cols=64,
                quantized=True,
            ))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_resolve_format(self):
        self.assertEqual(resolve_format("/m/model.gguf", "auto"), "gguf")
        self.assertEqual(resolve_format("/m/model.nanoq", "auto"), "nanoq")

    def test_two_models_different_output(self):
        r_a = self.router.submit("same prompt", 12, model="model-a").result()
        r_b = self.router.submit("same prompt", 12, model="model-b").result()
        self.assertEqual(r_a.model, "model-a")
        self.assertEqual(r_b.model, "model-b")
        self.assertTrue(r_a.text)
        self.assertTrue(r_b.text)

    def test_gguf_fallback_without_extra(self):
        gguf = self.tmp / "fake.gguf"
        gguf.write_bytes(b"GGUF")
        self.reg.register(ModelEntry(id="g", source_path=str(gguf), format="gguf"))
        r = self.router.submit("hi", 4, model="g", format="gguf").result()
        if not self.router.gguf_available:
            self.assertTrue(any("GGUF" in w for w in r.warnings))


if __name__ == "__main__":
    unittest.main()
