"""High-level NanoServe SDK client."""
from __future__ import annotations

import asyncio
from typing import Literal

from nanoserve.engine.pool import DeviceLiteral, EnginePool
from nanoserve.engine.worker import InferResult

DeviceOption = Literal["cpu", "gpu", "auto"]


class NanoServe:
    """In-process inference without HTTP."""

    def __init__(
        self,
        device: DeviceOption = "cpu",
        num_workers: int = 4,
        lib_path: str | None = None,
    ):
        self.default_device: DeviceOption = device
        self.pool = EnginePool(num_workers=num_workers, lib_path=lib_path)
        self.last_device: str = "cpu"
        self.last_warnings: list[str] = []

    def generate(
        self,
        prompt: str,
        max_tokens: int = 24,
        device: DeviceOption | None = None,
    ) -> str:
        result = self._run(prompt, max_tokens, device or self.default_device)
        return result.text

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 24,
        device: DeviceOption | None = None,
    ) -> str:
        result = await self._run_async(prompt, max_tokens, device or self.default_device)
        return result.text

    def _run(self, prompt: str, max_tokens: int, device: DeviceLiteral) -> InferResult:
        future = self.pool.submit(prompt, max_tokens, device=device)
        result: InferResult = future.result()
        self.last_device = result.device
        self.last_warnings = list(result.warnings)
        return result

    async def _run_async(self, prompt: str, max_tokens: int, device: DeviceLiteral) -> InferResult:
        loop = asyncio.get_event_loop()
        future = self.pool.submit(prompt, max_tokens, device=device)
        result: InferResult = await loop.run_in_executor(None, future.result)
        self.last_device = result.device
        self.last_warnings = list(result.warnings)
        return result
