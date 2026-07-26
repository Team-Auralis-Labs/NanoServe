"""Thread pool for optional GGUF inference."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from nanoserve.engine.gguf_worker import infer_gguf
from nanoserve.engine.worker import InferResult


class GGUFPool:
    def __init__(self, workers: int = 1):
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gguf")

    def submit(
        self,
        prompt: str,
        max_tokens: int,
        *,
        model_path: str,
        device: str,
    ):
        def run() -> InferResult:
            text, device_used, warnings = infer_gguf(model_path, prompt, max_tokens, device)
            return InferResult(
                text=text,
                device=device_used,
                warnings=warnings,
                format="gguf",
                model=model_path,
                quantized=True,
            )

        return self.executor.submit(run)
