"""Prepare raw weights into .nanoq or passthrough paths."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from nanoserve.models.registry import ModelEntry, ModelRegistry
from nanoserve.quantizer.quantize import Quantizer

RAW_WARNING = "Using unquantized weights; not recommended."


def _auto_quantize_default() -> bool:
    return os.environ.get("NANOSERVE_AUTO_QUANTIZE", "1") not in ("0", "false", "False")


def _default_precision() -> str:
    return os.environ.get("NANOSERVE_DEFAULT_PRECISION", "int8")


def detect_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".nanoq":
        return "nanoq"
    if ext == ".gguf":
        return "gguf"
    if ext in (".safetensors",):
        return "safetensors"
    if ext in (".bin", ".pt", ".pth"):
        return "bin"
    return "unknown"


def prepare_model(
    model: str | None,
    *,
    registry: ModelRegistry | None = None,
    quantize: bool | None = None,
    precision: str = "int8",
) -> tuple[str | None, str, ModelEntry | None, list[str], bool]:
    """Return (resolved_path, runtime_format, entry, warnings, quantized)."""
    reg = registry or ModelRegistry()
    warnings: list[str] = []
    do_quantize = _auto_quantize_default() if quantize is None else quantize
    prec = precision if precision != "int8" else _default_precision()

    resolved = reg.resolve_path(model)
    if not resolved:
        return None, "nanoq", None, warnings, True

    path = Path(resolved)
    fmt = detect_format(str(path))
    entry = reg.get(model) if model and model in reg._entries else None

    if fmt == "gguf":
        return str(path.resolve()), "gguf", entry, warnings, False

    if fmt == "nanoq":
        if entry:
            entry.nanoq_path = str(path.resolve())
            entry.quantized = path.suffix == ".nanoq"
            reg.register(entry)
        return str(path.resolve()), "nanoq", entry, warnings, True

    if fmt in ("safetensors", "bin", "unknown"):
        model_id = entry.id if entry else path.parent.name
        out_dir = reg.root / model_id
        out_dir.mkdir(parents=True, exist_ok=True)

        if not do_quantize or prec == "raw":
            warnings.append(RAW_WARNING)
            try:
                if fmt == "safetensors":
                    weights = Quantizer.load_safetensors_matrix(str(path))
                else:
                    weights = np.fromfile(path, dtype=np.float32)
                nanoq_path = out_dir / f"{model_id}-fp16.nanoq"
                Quantizer.from_weights(weights, str(nanoq_path), precision="fp16", name=model_id)
                if entry:
                    entry.nanoq_path = str(nanoq_path.resolve())
                    entry.dtype = "fp16"
                    entry.quantized = False
                    reg.register(entry)
                return str(nanoq_path.resolve()), "nanoq", entry, warnings, False
            except Exception as exc:
                warnings.append(f"Raw load failed: {exc}")
                return str(path.resolve()), "nanoq", entry, warnings, False

        try:
            if fmt == "safetensors":
                weights = Quantizer.load_safetensors_matrix(str(path))
            else:
                weights = np.fromfile(path, dtype=np.float32)
            if weights.ndim > 2:
                weights = weights.reshape(weights.shape[0], -1)
            nanoq_path = out_dir / f"{model_id}-{prec}.nanoq"
            Quantizer.from_weights(weights, str(nanoq_path), precision=prec, name=model_id)
            info = Quantizer.read_nanoq(str(nanoq_path))
            hdr = info["header"]
            if entry:
                entry.nanoq_path = str(nanoq_path.resolve())
                entry.dtype = hdr.get("dtype", prec)
                entry.rows = int(hdr.get("rows", 0))
                entry.cols = int(hdr.get("cols", 0))
                entry.quantized = prec in ("int8", "fp4")
                entry.format = "nanoq"
                reg.register(entry)
            return str(nanoq_path.resolve()), "nanoq", entry, warnings, prec in ("int8", "fp4")
        except Exception as exc:
            warnings.append(f"Quantization failed: {exc}; using source path")
            return str(path.resolve()), "nanoq", entry, warnings, False

    return str(path.resolve()), "nanoq", entry, warnings, True
