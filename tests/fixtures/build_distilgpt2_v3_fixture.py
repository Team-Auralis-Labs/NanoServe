"""Build distilgpt2 .nanoq v3 fixture for Phase 01 tests."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

NANOQ_V3_MAGIC = 0x4E515033
ALIGN = 64


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _blake3_footer(data: bytes) -> bytes:
    import blake3

    return blake3.blake3(data).digest()


def quantize_int8(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    scales = np.max(np.abs(weights), axis=-1, keepdims=True).astype(np.float32) / 127.0
    scales[scales == 0] = 1.0
    q = np.round(weights / scales).astype(np.int8)
    return q, scales.squeeze().astype(np.float32)


def write_nanoq_v3(
    path: Path,
    config: dict,
    tensors: dict[str, np.ndarray],
    tokenizer_bytes: bytes,
    quantize_large: bool = True,
) -> None:
    entries = []
    payload_parts: list[bytes] = []
    scale_parts: list[bytes] = []
    offset = 0
    scale_offset_base = 0

    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        use_int8 = (
            quantize_large
            and arr.ndim == 2
            and arr.size > 4096
            and not name.endswith("wte.weight")
            and not name.endswith("wpe.weight")
            and not name.endswith("lm_head.weight")
        )
        if use_int8:
            q, scales = quantize_int8(arr.astype(np.float32))
            raw = q.tobytes()
            scale_raw = scales.tobytes()
            dtype = "int8"
            quant = "per-row"
            shape = list(q.shape)
            scale_off = _align(offset + len(raw))
        else:
            raw = arr.astype(np.float32).tobytes()
            scale_raw = b""
            dtype = "fp32"
            quant = "none"
            shape = list(arr.shape)
            scale_off = 0

        aligned_off = _align(offset)
        if aligned_off > offset:
            payload_parts.append(b"\x00" * (aligned_off - offset))
            offset = aligned_off

        entry = {
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "offset": offset,
            "size": len(raw),
            "scale_offset": scale_off if use_int8 else 0,
            "quant": quant,
            "block_size": 32,
        }
        entries.append(entry)
        payload_parts.append(raw)
        offset += len(raw)
        if use_int8:
            scale_parts.append((scale_off, scale_raw))

    for scale_off, scale_raw in scale_parts:
        if scale_off > offset:
            payload_parts.append(b"\x00" * (scale_off - offset))
            offset = scale_off
        payload_parts.append(scale_raw)
        offset += len(scale_raw)

    payload = b"".join(payload_parts)
    index_json = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    config_json = json.dumps(config, separators=(",", ":")).encode("utf-8")

    header = b""
    header += struct.pack("<I", NANOQ_V3_MAGIC)
    header += struct.pack("<I", len(index_json))
    header += index_json
    header += struct.pack("<I", len(config_json))
    header += config_json
    header += struct.pack("<I", len(tokenizer_bytes))
    header += tokenizer_bytes
    pad = _align(len(header)) - len(header)
    header += b"\x00" * pad

    body = header + payload
    footer = _blake3_footer(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body + footer)


def export_distilgpt2(out_path: Path, quantize_large: bool = False) -> None:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit("Install transformers: pip install transformers torch") from e

    model_name = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    state = model.state_dict()

    tensors: dict[str, np.ndarray] = {}
    for key, val in state.items():
        arr = val.detach().cpu().numpy()
        # HF GPT-2 Conv1D stores [in, out]; engine expects [out, in]
        if (
            arr.ndim == 2
            and not key.endswith("wte.weight")
            and not key.endswith("wpe.weight")
            and not key.endswith("lm_head.weight")
        ):
            arr = arr.T.copy()
        tensors[key] = arr

    config = {
        "arch": "gpt2",
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.n_embd,
        "n_layers": model.config.n_layer,
        "n_heads": model.config.n_head,
        "n_kv_heads": model.config.n_head,
        "max_seq_len": model.config.n_positions,
        "norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "act_fn": "gelu",
    }

    tok_dir = Path("/tmp/_tok_export_distilgpt2")
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(tok_dir))
    tokenizer_bytes = (tok_dir / "tokenizer.json").read_bytes()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_nanoq_v3(out_path, config, tensors, tokenizer_bytes, quantize_large=quantize_large)
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(ROOT / "tests" / "fixtures" / "distilgpt2-int8.nanoq"),
    )
    parser.add_argument(
        "--precision",
        choices=("int8", "fp32"),
        default="fp32",
    )
    args = parser.parse_args()
    export_distilgpt2(Path(args.out), quantize_large=(args.precision == "int8"))
