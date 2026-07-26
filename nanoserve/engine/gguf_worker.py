"""Lazy shared GGUF model via llama-cpp-python."""
from __future__ import annotations

import os
import threading
from typing import Any

_lock = threading.Lock()
_model: Any = None
_model_path: str | None = None
_load_key: tuple[str, str] | None = None


def default_gguf_path() -> str | None:
    env = os.environ.get("NANOSERVE_MODEL_PATH")
    if env and env.endswith(".gguf") and os.path.isfile(env):
        return env
    return None


def _n_ctx() -> int:
    return int(os.environ.get("NANOSERVE_GGUF_N_CTX", "2048"))


def _n_batch() -> int:
    return int(os.environ.get("NANOSERVE_GGUF_N_BATCH", "512"))


def _n_threads() -> int:
    val = int(os.environ.get("NANOSERVE_GGUF_N_THREADS", "0"))
    if val <= 0:
        return os.cpu_count() or 4
    return val


def _n_gpu_layers(device: str) -> int:
    if device in ("gpu", "auto"):
        return int(os.environ.get("NANOSERVE_GGUF_N_GPU_LAYERS", "0"))
    return 0


def active_format() -> str:
    return "gguf" if _model is not None else "nanoq"


def load_gguf(model_path: str, device: str = "cpu") -> Any:
    global _model, _model_path, _load_key
    key = (model_path, device)
    with _lock:
        if _model is not None and _load_key == key:
            return _model
        from llama_cpp import Llama

        if _model is not None:
            try:
                del _model
            except Exception:
                pass
            _model = None

        _model = Llama(
            model_path=model_path,
            n_ctx=_n_ctx(),
            n_batch=_n_batch(),
            n_threads=_n_threads(),
            n_gpu_layers=_n_gpu_layers(device),
            verbose=False,
        )
        _model_path = model_path
        _load_key = key
        return _model


def infer_gguf(model_path: str, prompt: str, max_tokens: int, device: str) -> tuple[str, str, list[str]]:
    """Return (text, device_used, warnings)."""
    warnings: list[str] = []
    layers = _n_gpu_layers(device)
    if device in ("gpu", "auto") and layers == 0:
        warnings.append("GGUF GPU requested but NANOSERVE_GGUF_N_GPU_LAYERS=0; using CPU")
    llm = load_gguf(model_path, device)
    out = llm(prompt, max_tokens=max_tokens, echo=False)
    device_used = "gpu" if layers > 0 and device in ("gpu", "auto") else "cpu"
    if isinstance(out, dict):
        choices = out.get("choices") or []
        if choices:
            text = choices[0].get("text") or choices[0].get("message", {}).get("content", "")
            return str(text).strip(), device_used, warnings
    return str(out), device_used, warnings


def gguf_model_loaded() -> bool:
    return _model is not None
