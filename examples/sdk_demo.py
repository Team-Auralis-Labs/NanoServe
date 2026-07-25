#!/usr/bin/env python3
"""Standalone SDK demo — quantization and in-process inference."""
import numpy as np

from nanoserve import NanoServe, Quantizer


def main() -> None:
    print("[*] Quantizer demo")
    w = np.random.randn(256, 1024).astype(np.float32)
    q, scales = Quantizer.quantize_int8(w)
    print(f"    quantized shape={q.shape}, scales shape={scales.shape}")

    print("[*] Inference demo")
    for device in ("cpu", "gpu", "auto"):
        engine = NanoServe(device=device)
        text = engine.generate("Hello world", max_tokens=50)
        print(f"    requested={device} used={engine.last_device} warnings={engine.last_warnings}")
        print(f"    output: {text[:120]}{'...' if len(text) > 120 else ''}")


if __name__ == "__main__":
    main()
