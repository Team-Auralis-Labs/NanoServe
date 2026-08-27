#!/usr/bin/env python3
"""Tokenizer parity: Rust FFI vs transformers."""
from __future__ import annotations

import ctypes
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNTIME_LIB = ROOT / "engine" / "build" / "libnanoserve_engine.so"
FIXTURE = ROOT / "tests" / "fixtures" / "distilgpt2-int8.nanoq"


def _read_tokenizer_blob() -> bytes:
    data = FIXTURE.read_bytes()
    magic = int.from_bytes(data[0:4], "little")
    assert magic == 0x4E515033
    index_len = int.from_bytes(data[4:8], "little")
    index_end = 8 + index_len
    config_len = int.from_bytes(data[index_end : index_end + 4], "little")
    config_end = index_end + 4 + config_len
    tok_len = int.from_bytes(data[config_end : config_end + 4], "little")
    tok_start = config_end + 4
    return data[tok_start : tok_start + tok_len]


class TestTokenizerRust(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RUNTIME_LIB.exists():
            raise unittest.SkipTest("nanoq_runtime not built")
        if not FIXTURE.exists():
            raise unittest.SkipTest("Fixture missing")
        cls.lib = ctypes.CDLL(str(RUNTIME_LIB))
        cls.lib.nanoq_tokenizer_create.restype = ctypes.c_void_p
        cls.lib.nanoq_tokenizer_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        cls.lib.nanoq_tokenizer_destroy.argtypes = [ctypes.c_void_p]
        cls.lib.nanoq_tokenizer_encode.restype = ctypes.c_int
        cls.lib.nanoq_tokenizer_encode.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
        ]
        cls.lib.nanoq_tokenizer_decode.restype = ctypes.c_void_p
        cls.lib.nanoq_tokenizer_decode.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
        ]
        cls.lib.nanoq_string_free.argtypes = [ctypes.c_char_p]

        blob = _read_tokenizer_blob()
        cls.handle = cls.lib.nanoq_tokenizer_create(blob, len(blob))
        if not cls.handle:
            raise unittest.SkipTest("tokenizer create failed")

    @classmethod
    def tearDownClass(cls):
        # Avoid allocator teardown races between Rust tokenizers and Python exit.
        cls.handle = None

    def _encode_rust(self, text: str) -> list[int]:
        buf = (ctypes.c_uint32 * 256)()
        n = self.lib.nanoq_tokenizer_encode(
            self.handle, text.encode("utf-8"), buf, 256
        )
        return [int(buf[i]) for i in range(n)]

    def _decode_rust(self, ids: list[int]) -> str:
        arr = (ctypes.c_uint32 * len(ids))(*ids)
        ptr = self.lib.nanoq_tokenizer_decode(self.handle, arr, len(ids))
        if not ptr:
            return ""
        out = ctypes.string_at(ptr).decode("utf-8")
        self.lib.nanoq_string_free(ctypes.c_char_p(ptr))
        return out

    def test_roundtrip(self):
        text = "Hello, NanoServe tokenizer parity check."
        ids = self._encode_rust(text)
        self.assertTrue(ids)
        back = self._decode_rust(ids)
        self.assertEqual(back, text)

    def test_matches_transformers(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("transformers not installed")
        hf = AutoTokenizer.from_pretrained("distilgpt2")
        samples = ["Hello world", "The quick brown fox", "NanoServe Phase 01"]
        for s in samples:
            rust_ids = self._encode_rust(s)
            hf_ids = hf.encode(s, add_special_tokens=False)
            self.assertEqual(rust_ids, hf_ids, msg=s)


if __name__ == "__main__":
    unittest.main()
