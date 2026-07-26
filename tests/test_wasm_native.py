#!/usr/bin/env python3
"""Native tests for WASM-oriented buffer FFI (runs against libnanoserve_engine.so)."""
from __future__ import annotations

import ctypes
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LD_LIBRARY_PATH", str(ROOT / "allocator" / "target" / "release"))
LIB = os.environ.get(
    "NANOSERVE_ENGINE_LIB", str(ROOT / "engine" / "build" / "libnanoserve_engine.so")
)

from nanoserve import Quantizer


class TestWasmBufferFFI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(LIB).exists():
            raise unittest.SkipTest("Engine not built")
        cls.lib = ctypes.CDLL(LIB)
        cls.lib.engine_init_with_model_bytes.restype = ctypes.c_void_p
        cls.lib.engine_init_with_model_bytes.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        cls.lib.engine_reload_model_bytes.restype = ctypes.c_int
        cls.lib.engine_reload_model_bytes.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        cls.lib.engine_infer.restype = ctypes.c_int
        cls.lib.engine_infer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        cls.lib.engine_model_info.restype = ctypes.c_char_p
        cls.lib.engine_model_info.argtypes = [ctypes.c_void_p]
        cls.lib.engine_cleanup.argtypes = [ctypes.c_void_p]

    def _nanoq_bytes(self, precision: str) -> bytes:
        path = Path(f"/tmp/nanoserve_wasm_{precision}.nanoq")
        w = np.random.default_rng(9).standard_normal((32, 64)).astype(np.float32)
        Quantizer.from_weights(w, str(path), precision=precision, name="wasm-test")
        return path.read_bytes()

    def _infer(self, handle, prompt: str = "hello wasm", n: int = 8) -> str:
        buf = ctypes.create_string_buffer(4096)
        self.lib.engine_infer(handle, prompt.encode(), n, buf, len(buf))
        return buf.value.decode()

    def test_init_with_model_bytes_int8(self):
        data = self._nanoq_bytes("int8")
        arr = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        h = self.lib.engine_init_with_model_bytes(arr, len(data), 0)
        self.assertTrue(h)
        meta = self.lib.engine_model_info(h).decode()
        self.assertIn("int8", meta)
        out = self._infer(h)
        self.assertTrue(out)
        self.lib.engine_cleanup(h)

    def test_reload_model_bytes(self):
        d1 = self._nanoq_bytes("int8")
        d2 = self._nanoq_bytes("fp16")
        a1 = (ctypes.c_uint8 * len(d1)).from_buffer_copy(d1)
        h = self.lib.engine_init_with_model_bytes(a1, len(d1), 0)
        self.assertTrue(h)
        a2 = (ctypes.c_uint8 * len(d2)).from_buffer_copy(d2)
        rc = self.lib.engine_reload_model_bytes(h, a2, len(d2))
        self.assertEqual(rc, 0)
        meta = self.lib.engine_model_info(h).decode()
        self.assertIn("fp16", meta)
        self.lib.engine_cleanup(h)

    def test_parity_file_vs_buffer(self):
        path = Path("/tmp/nanoserve_wasm_parity.nanoq")
        w = np.random.default_rng(3).standard_normal((32, 64)).astype(np.float32)
        Quantizer.from_weights(w, str(path), precision="int8", name="parity")
        data = path.read_bytes()
        arr = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)

        self.lib.engine_init_with_model.restype = ctypes.c_void_p
        self.lib.engine_init_with_model.argtypes = [ctypes.c_char_p, ctypes.c_int]
        hf = self.lib.engine_init_with_model(str(path).encode(), 0)
        hb = self.lib.engine_init_with_model_bytes(arr, len(data), 0)
        self.assertTrue(hf and hb)
        out_f = self._infer(hf, "parity", 8)
        out_b = self._infer(hb, "parity", 8)
        self.assertEqual(out_f, out_b)
        self.lib.engine_cleanup(hf)
        self.lib.engine_cleanup(hb)


if __name__ == "__main__":
    unittest.main()
