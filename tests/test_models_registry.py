#!/usr/bin/env python3
"""Model registry tests."""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoserve.models.registry import ModelEntry, ModelRegistry


class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.reg = ModelRegistry(root=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_register_and_list(self):
        entry = ModelEntry(
            id="demo",
            source_path=str(self.tmp / "demo.safetensors"),
            format="safetensors",
        )
        self.reg.register(entry)
        self.assertEqual(self.reg.count, 1)
        self.assertEqual(self.reg.get("demo").id, "demo")

    def test_resolve_path(self):
        p = self.tmp / "weights.nanoq"
        p.write_bytes(b"\x00")
        entry = ModelEntry(id="w", source_path=str(p), nanoq_path=str(p))
        self.reg.register(entry)
        self.assertEqual(self.reg.resolve_path("w"), str(p.resolve()))

    def test_delete(self):
        entry = ModelEntry(id="x", source_path="/tmp/x.bin")
        self.reg.register(entry)
        self.assertTrue(self.reg.delete("x"))
        self.assertIsNone(self.reg.get("x"))


if __name__ == "__main__":
    unittest.main()
