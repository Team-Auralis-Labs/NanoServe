#!/usr/bin/env python3
"""Tests for fp16/fp4 quantizer."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoserve import Quantizer


class TestQuantizerFp16Fp4(unittest.TestCase):
    def test_fp16_roundtrip(self):
        w = np.random.randn(32, 64).astype(np.float32)
        path = Path("/tmp/nanoserve_q_fp16.nanoq")
        Quantizer.from_weights(w, str(path), precision="fp16")
        info = Quantizer.read_nanoq(str(path))
        self.assertEqual(info["header"]["dtype"], "fp16")
        self.assertEqual(info["weights"].shape, (32, 64))

    def test_fp4_roundtrip(self):
        w = np.random.randn(64, 32).astype(np.float32)
        path = Path("/tmp/nanoserve_q_fp4.nanoq")
        Quantizer.from_weights(w, str(path), precision="fp4")
        info = Quantizer.read_nanoq(str(path))
        self.assertEqual(info["header"]["dtype"], "fp4")
        self.assertGreater(len(info["scales"]), 0)

    def test_v1_compat_header(self):
        w = np.random.randn(8, 16).astype(np.float32)
        q, scales = Quantizer.quantize_int8(w)
        path = Path("/tmp/nanoserve_q_v1.nanoq")
        Quantizer.write_nanoq(str(path), q, scales=scales, rows=8, cols=16, dtype="int8")
        info = Quantizer.read_nanoq(str(path))
        self.assertEqual(info["header"].get("version", 1), 2)


if __name__ == "__main__":
    unittest.main()
