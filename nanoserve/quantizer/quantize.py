"""Minimal INT8 post-training quantizer.

Takes a float32 weight matrix and produces a per-channel INT8 quantized
version + scale factors, written to a .nanoq file the engine can read.
"""
from __future__ import annotations

import argparse
import json
import struct

import numpy as np


class Quantizer:
    @staticmethod
    def quantize_int8(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scales = np.max(np.abs(weights), axis=-1, keepdims=True) / 127.0
        scales[scales == 0] = 1.0
        q = np.round(weights / scales).astype(np.int8)
        return q, scales.astype(np.float32)

    @classmethod
    def write_nanoq(
        cls,
        path: str,
        q: np.ndarray,
        scales: np.ndarray,
        rows: int,
        cols: int,
    ) -> None:
        with open(path, "wb") as f:
            header = json.dumps({"rows": rows, "cols": cols, "dtype": "int8"}).encode()
            f.write(struct.pack("<I", len(header)))
            f.write(header)
            f.write(q.tobytes())
            f.write(scales.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description="NanoServe INT8 quantizer")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--out", default="model-int8.nanoq")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    weights = rng.standard_normal((args.rows, args.cols)).astype(np.float32)
    q, scales = Quantizer.quantize_int8(weights)
    Quantizer.write_nanoq(args.out, q, scales, args.rows, args.cols)
    print(
        f"[+] wrote {args.out}: {q.nbytes/1e6:.2f}MB int8 + {scales.nbytes} bytes scales "
        f"(vs {weights.nbytes/1e6:.2f}MB fp32 -> {weights.nbytes/q.nbytes:.1f}x smaller)"
    )


if __name__ == "__main__":
    main()
