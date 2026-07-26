"""Route inference to native .nanoq or optional GGUF runtime."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from nanoserve.engine.gguf_probe import gguf_available
from nanoserve.engine.gguf_pool import GGUFPool
from nanoserve.engine.gguf_worker import active_format, default_gguf_path
from nanoserve.engine.pool import DeviceLiteral, EnginePool
from nanoserve.engine.worker import InferResult
from nanoserve.models.cache import ModelCache
from nanoserve.models.pipeline import detect_format, prepare_model
from nanoserve.models.registry import ModelRegistry

FormatLiteral = Literal["auto", "nanoq", "gguf"]
PrecisionLiteral = Literal["int8", "fp16", "fp4", "raw"]


def resolve_format(path: str | None, fmt: FormatLiteral) -> FormatLiteral:
    if fmt != "auto":
        return fmt
    if path and detect_format(path) == "gguf":
        return "gguf"
    return "nanoq"


def _resolve_gguf_path(model: str | None, registry: ModelRegistry) -> str | None:
    if model:
        resolved = registry.resolve_path(model)
        if resolved:
            return resolved
    default = default_gguf_path()
    if default:
        return default
    env = os.environ.get("NANOSERVE_MODEL_PATH")
    return registry.resolve_path(env) if env else None


class InferenceRouter:
    def __init__(
        self,
        num_workers: int = 4,
        lib_path: str | None = None,
        registry: ModelRegistry | None = None,
    ):
        self.registry = registry or ModelRegistry()
        self.native_pool = EnginePool(num_workers=num_workers, lib_path=lib_path)
        self.gguf_pool = GGUFPool() if gguf_available() else None
        self.model_cache = ModelCache()

    @property
    def gpu_cuda_available(self) -> bool:
        return self.native_pool.gpu_cuda_available

    @property
    def gpu_opencl_available(self) -> bool:
        return self.native_pool.gpu_opencl_available

    @property
    def gpu_available(self) -> bool:
        return self.native_pool.gpu_available

    @property
    def gguf_available(self) -> bool:
        return gguf_available()

    def submit(
        self,
        prompt: str,
        max_tokens: int,
        *,
        device: DeviceLiteral = "cpu",
        model: str | None = None,
        format: FormatLiteral = "auto",
        quantize: bool | None = None,
        precision: PrecisionLiteral = "int8",
    ):
        requested_format = format
        if requested_format == "gguf" or (
            requested_format == "auto"
            and os.environ.get("NANOSERVE_DEFAULT_FORMAT", "auto") == "gguf"
        ):
            gguf_path = _resolve_gguf_path(model, self.registry)
            if gguf_path and Path(gguf_path).suffix.lower() == ".gguf":
                return self._gguf_submit(
                    prompt, max_tokens, device=device, model_path=gguf_path,
                    model_label=model or gguf_path, warnings=[],
                )

        path, runtime_fmt, entry, prep_warnings, quantized = prepare_model(
            model,
            registry=self.registry,
            quantize=quantize,
            precision=precision,
        )
        use_format = resolve_format(path, requested_format if requested_format != "auto" else runtime_fmt)  # type: ignore[arg-type]

        warnings = list(prep_warnings)
        model_label = model or (entry.id if entry else None) or path

        if use_format == "gguf":
            if not path or detect_format(path) != "gguf":
                warnings.append("GGUF format requested but no .gguf model found; falling back to native")
                return self._native_submit(
                    prompt, max_tokens, device=device, model_path=None,
                    warnings=warnings, model_label=model_label, quantized=quantized,
                )
            return self._gguf_submit(
                prompt, max_tokens, device=device, model_path=path,
                model_label=model_label, warnings=warnings,
            )

        return self._native_submit(
            prompt, max_tokens, device=device, model_path=path,
            warnings=warnings, model_label=model_label, quantized=quantized,
        )

    def _gguf_submit(
        self,
        prompt: str,
        max_tokens: int,
        *,
        device: DeviceLiteral,
        model_path: str,
        model_label: str | None,
        warnings: list[str],
    ):
        if not self.gguf_pool:
            warnings.append("GGUF runtime unavailable; falling back to native")
            return self._native_submit(
                prompt, max_tokens, device=device, model_path=None,
                warnings=warnings, model_label=model_label, quantized=False,
            )
        fut = self.gguf_pool.submit(
            prompt, max_tokens, model_path=model_path, device=device,
        )

        def wrap_gguf():
            result = fut.result()
            result.model = model_label or model_path
            result.warnings = warnings + result.warnings
            return result

        return self.native_pool.cpu_executor.submit(wrap_gguf)

    def _native_submit(
        self,
        prompt: str,
        max_tokens: int,
        *,
        device: DeviceLiteral,
        model_path: str | None,
        warnings: list[str],
        model_label: str | None,
        quantized: bool,
    ):
        if model_path:
            self.model_cache.touch(model_path, device)

        fut = self.native_pool.submit(
            prompt, max_tokens, device=device, model_path=model_path,
        )

        def wrap_native():
            result = fut.result()
            result.format = "nanoq"
            result.model = model_label
            result.quantized = quantized
            result.warnings = warnings + result.warnings
            return result

        return self.native_pool.cpu_executor.submit(wrap_native)

    def active_format(self) -> str:
        if self.gguf_pool and active_format() == "gguf":
            return "gguf"
        return "nanoq"
