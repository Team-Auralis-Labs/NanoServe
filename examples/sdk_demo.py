#!/usr/bin/env python3
"""Standalone SDK demo — quantization, models, and in-process inference."""
import numpy as np
from pathlib import Path

from nanoserve import NanoServe, Quantizer


def main() -> None:
    print("[*] Quantizer demo (.nanoq v2)")
    w = np.random.randn(64, 128).astype(np.float32)
    out = Path("/tmp/nanoserve_demo.nanoq")
    Quantizer.from_weights(w, str(out), precision="int8", name="demo")
    print(f"    wrote {out}")

    print("[*] Inference demo")
    engine = NanoServe(device="auto", model=str(out), format="nanoq")
    text = engine.generate("Hello world", max_tokens=24)
    print(f"    model={engine.last_model} format={engine.last_format} quantized={engine.last_quantized}")
    print(f"    device={engine.last_device} warnings={engine.last_warnings}")
    print(f"    output: {text[:120]}{'...' if len(text) > 120 else ''}")
    print(f"    registered models: {[m['id'] for m in engine.list_models()]}")


if __name__ == "__main__":
    main()
