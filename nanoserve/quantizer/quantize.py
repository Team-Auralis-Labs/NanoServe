"""Minimal post-training quantizer for .nanoq v2 format.

Supports int8, fp16, and fp4 dtypes. Raw safetensors can be loaded optionally.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

PrecisionLiteral = str  # int8 | fp16 | fp4 | raw


class Quantizer:
    @staticmethod
    def quantize_int8(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scales = np.max(np.abs(weights), axis=-1, keepdims=True) / 127.0
        scales[scales == 0] = 1.0
        q = np.round(weights / scales).astype(np.int8)
        return q, scales.astype(np.float32)

    @staticmethod
    def quantize_fp16(weights: np.ndarray) -> np.ndarray:
        return weights.astype(np.float16)

    @staticmethod
    def quantize_fp4(weights: np.ndarray, block_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        flat = weights.reshape(-1).astype(np.float32)
        n = flat.size
        num_blocks = (n + block_size - 1) // block_size
        scales = np.zeros(num_blocks, dtype=np.float32)
        packed = np.zeros((n + 1) // 2, dtype=np.uint8)
        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, n)
            block = flat[start:end]
            scale = float(np.max(np.abs(block))) if block.size else 1.0
            if scale == 0:
                scale = 1.0
            scales[b] = scale
            q = np.clip(np.round(block / scale * 7.0), -8, 7).astype(np.int8)
            for i, val in enumerate(q):
                idx = start + i
                nibble = int(val) & 0x0F
                if idx % 2 == 0:
                    packed[idx // 2] = nibble
                else:
                    packed[idx // 2] |= nibble << 4
        return packed, scales

    @classmethod
    def write_nanoq(
        cls,
        path: str,
        weights_or_q: np.ndarray,
        scales: np.ndarray | None = None,
        rows: int | None = None,
        cols: int | None = None,
        dtype: str = "int8",
        block_size: int = 32,
        name: str = "",
    ) -> None:
        if rows is None or cols is None:
            if weights_or_q.ndim == 1:
                rows, cols = 1, weights_or_q.size
            else:
                rows, cols = weights_or_q.shape

        header: dict[str, Any] = {
            "version": 2,
            "rows": int(rows),
            "cols": int(cols),
            "dtype": dtype,
            "block_size": block_size,
        }
        if name:
            header["name"] = name

        with open(path, "wb") as f:
            header_bytes = json.dumps(header, separators=(",", ":")).encode()
            f.write(struct.pack("<I", len(header_bytes)))
            f.write(header_bytes)

            if dtype == "int8":
                q = weights_or_q.astype(np.int8)
                sc = scales if scales is not None else np.ones((rows, 1), dtype=np.float32)
                f.write(q.tobytes())
                f.write(sc.astype(np.float32).tobytes())
            elif dtype == "fp16":
                fp16 = weights_or_q.astype(np.float16)
                f.write(fp16.tobytes())
            elif dtype == "fp4":
                packed = weights_or_q.astype(np.uint8)
                sc = scales if scales is not None else np.ones(1, dtype=np.float32)
                f.write(packed.tobytes())
                f.write(sc.astype(np.float32).tobytes())
            else:
                raise ValueError(f"unsupported dtype: {dtype}")

    @classmethod
    def read_nanoq(cls, path: str) -> dict[str, Any]:
        with open(path, "rb") as f:
            header_len = struct.unpack("<I", f.read(4))[0]
            header = json.loads(f.read(header_len).decode())
            rows = int(header["rows"])
            cols = int(header["cols"])
            dtype = header.get("dtype", "int8")
            block_size = int(header.get("block_size", 32))
            elements = rows * cols

            if dtype == "int8":
                q = np.frombuffer(f.read(elements), dtype=np.int8).reshape(rows, cols)
                scales = np.frombuffer(f.read(rows * 4), dtype=np.float32).reshape(rows, 1)
                return {"header": header, "weights": q, "scales": scales}
            if dtype == "fp16":
                w = np.frombuffer(f.read(elements * 2), dtype=np.float16).reshape(rows, cols)
                return {"header": header, "weights": w}
            if dtype == "fp4":
                packed = np.frombuffer(f.read((elements + 1) // 2), dtype=np.uint8)
                num_blocks = (elements + block_size - 1) // block_size
                scales = np.frombuffer(f.read(num_blocks * 4), dtype=np.float32)
                return {"header": header, "weights": packed, "scales": scales}
            raise ValueError(f"unsupported dtype: {dtype}")

    @classmethod
    def from_weights(
        cls,
        weights: np.ndarray,
        path: str,
        *,
        precision: str = "int8",
        block_size: int = 32,
        name: str = "",
    ) -> None:
        if weights.ndim == 1:
            rows, cols = 1, weights.size
        else:
            rows, cols = weights.shape

        if precision == "int8":
            q, scales = cls.quantize_int8(weights)
            cls.write_nanoq(path, q, scales=scales, rows=rows, cols=cols, dtype="int8", name=name)
        elif precision in ("fp16", "raw"):
            fp16 = cls.quantize_fp16(weights)
            cls.write_nanoq(path, fp16, rows=rows, cols=cols, dtype="fp16", name=name)
        elif precision == "fp4":
            packed, scales = cls.quantize_fp4(weights, block_size=block_size)
            cls.write_nanoq(
                path, packed, scales=scales, rows=rows, cols=cols,
                dtype="fp4", block_size=block_size, name=name,
            )
        else:
            raise ValueError(f"unsupported precision: {precision}")

    @classmethod
    def load_safetensors_matrix(cls, path: str, key: str | None = None) -> np.ndarray:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError("safetensors required: pip install safetensors") from exc

        with safe_open(path, framework="numpy") as st:
            keys = list(st.keys())
            if not keys:
                raise ValueError("empty safetensors file")
            use_key = key or keys[0]
            arr = st.get_tensor(use_key)
            return np.asarray(arr, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="NanoServe quantizer (.nanoq v2)")
    ap.add_argument("--input", help="Input safetensors or omit for random demo weights")
    ap.add_argument("--key", help="Tensor key inside safetensors")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--cols", type=int, default=1024)
    ap.add_argument("--out", default="model-int8.nanoq")
    ap.add_argument("--precision", choices=["int8", "fp16", "fp4", "raw"], default="int8")
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    if args.input:
        weights = Quantizer.load_safetensors_matrix(args.input, args.key)
        if weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)
    else:
        rng = np.random.default_rng(0)
        weights = rng.standard_normal((args.rows, args.cols)).astype(np.float32)

    Quantizer.from_weights(weights, args.out, precision=args.precision, name=args.name or Path(args.out).stem)
    print(f"[+] wrote {args.out} precision={args.precision}")


if __name__ == "__main__":
    main()
