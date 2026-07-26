"""High-level NanoServe SDK client."""
from __future__ import annotations

import asyncio
from typing import Literal

from nanoserve.engine.pool import DeviceLiteral
from nanoserve.engine.router import FormatLiteral, InferenceRouter, PrecisionLiteral
from nanoserve.engine.worker import InferResult
from nanoserve.models.download import download_model
from nanoserve.models.registry import ModelRegistry

DeviceOption = Literal["cpu", "gpu", "auto"]


class NanoServe:
    """In-process inference without HTTP."""

    def __init__(
        self,
        device: DeviceOption = "cpu",
        num_workers: int = 4,
        lib_path: str | None = None,
        model: str | None = None,
        format: FormatLiteral = "auto",
        registry: ModelRegistry | None = None,
    ):
        self.default_device: DeviceOption = device
        self.default_model = model
        self.default_format: FormatLiteral = format
        self.registry = registry or ModelRegistry()
        self.router = InferenceRouter(
            num_workers=num_workers, lib_path=lib_path, registry=self.registry,
        )
        self.last_device: str = "cpu"
        self.last_warnings: list[str] = []
        self.last_format: str = "nanoq"
        self.last_model: str | None = None
        self.last_quantized: bool = True

    def list_models(self) -> list[dict]:
        return [e.to_dict() for e in self.registry.list()]

    def download(
        self,
        source: str,
        *,
        model_id: str | None = None,
        repo_id: str | None = None,
        url: str | None = None,
        filename: str | None = None,
        revision: str | None = None,
        quantize: bool | None = None,
        precision: PrecisionLiteral = "int8",
    ) -> dict:
        entry = download_model(
            source=source,
            model_id=model_id,
            repo_id=repo_id,
            url=url,
            filename=filename,
            revision=revision,
            registry=self.registry,
        )
        from nanoserve.models.pipeline import prepare_model

        path, fmt, updated, warnings, quantized = prepare_model(
            entry.id, registry=self.registry, quantize=quantize, precision=precision,
        )
        return {
            "entry": (updated or entry).to_dict(),
            "resolved_path": path,
            "format": fmt,
            "quantized": quantized,
            "warnings": warnings,
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 24,
        device: DeviceOption | None = None,
        model: str | None = None,
        format: FormatLiteral | None = None,
        quantize: bool | None = None,
        precision: PrecisionLiteral = "int8",
    ) -> str:
        result = self._run(
            prompt, max_tokens,
            device or self.default_device,
            model or self.default_model,
            format or self.default_format,
            quantize,
            precision,
        )
        return result.text

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 24,
        device: DeviceOption | None = None,
        model: str | None = None,
        format: FormatLiteral | None = None,
        quantize: bool | None = None,
        precision: PrecisionLiteral = "int8",
    ) -> str:
        result = await self._run_async(
            prompt, max_tokens,
            device or self.default_device,
            model or self.default_model,
            format or self.default_format,
            quantize,
            precision,
        )
        return result.text

    def _apply_result(self, result: InferResult) -> InferResult:
        self.last_device = result.device
        self.last_warnings = list(result.warnings)
        self.last_format = result.format
        self.last_model = result.model
        self.last_quantized = result.quantized
        return result

    def _run(
        self,
        prompt: str,
        max_tokens: int,
        device: DeviceLiteral,
        model: str | None,
        format: FormatLiteral,
        quantize: bool | None,
        precision: PrecisionLiteral,
    ) -> InferResult:
        future = self.router.submit(
            prompt, max_tokens, device=device, model=model,
            format=format, quantize=quantize, precision=precision,
        )
        return self._apply_result(future.result())

    async def _run_async(
        self,
        prompt: str,
        max_tokens: int,
        device: DeviceLiteral,
        model: str | None,
        format: FormatLiteral,
        quantize: bool | None,
        precision: PrecisionLiteral,
    ) -> InferResult:
        loop = asyncio.get_event_loop()
        future = self.router.submit(
            prompt, max_tokens, device=device, model=model,
            format=format, quantize=quantize, precision=precision,
        )
        result: InferResult = await loop.run_in_executor(None, future.result)
        return self._apply_result(result)
