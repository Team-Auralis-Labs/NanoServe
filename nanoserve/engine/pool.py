"""Thread pool with CPU and GPU worker routing."""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from nanoserve.engine.worker import BackendKind, EngineWorker, InferResult

logger = logging.getLogger(__name__)

DeviceLiteral = Literal["cpu", "gpu", "auto"]


@dataclass
class _GpuState:
    cuda: bool
    opencl: bool
    backend: BackendKind | None
    backend_name: str

    @property
    def available(self) -> bool:
        return self.cuda or self.opencl


class EnginePool:
    """Fixed-size pools for CPU and optional GPU backends."""

    def __init__(
        self,
        num_workers: int = 4,
        gpu_workers: int | None = None,
        lib_path: str | None = None,
    ):
        self.lib_path = lib_path
        self.gpu_workers = gpu_workers or int(os.environ.get("NANOSERVE_GPU_WORKERS", "1"))
        self._gpu_state = self._detect_gpu()
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="engine-cpu"
        )
        self.gpu_executor = (
            ThreadPoolExecutor(max_workers=self.gpu_workers, thread_name_prefix="engine-gpu")
            if self._gpu_state.available
            else None
        )
        self._cpu_workers: dict[tuple[str | None, str], EngineWorker] = {}
        self._gpu_workers_map: dict[tuple[str | None, str], EngineWorker] = {}
        self._cpu_lock = threading.Lock()
        self._gpu_lock = threading.Lock()

    def _detect_gpu(self) -> _GpuState:
        cuda = EngineWorker.probe_cuda(self.lib_path)
        ocl = EngineWorker.probe_opencl(self.lib_path)
        if cuda:
            return _GpuState(True, ocl, BackendKind.CUDA, "cuda")
        if ocl:
            return _GpuState(False, True, BackendKind.OPENCL, "opencl")
        return _GpuState(False, False, None, "cpu")

    @property
    def gpu_cuda_available(self) -> bool:
        return self._gpu_state.cuda

    @property
    def gpu_opencl_available(self) -> bool:
        return self._gpu_state.opencl

    @property
    def gpu_available(self) -> bool:
        return self._gpu_state.available

    def _get_worker(
        self,
        model_path: str | None,
        backend: BackendKind,
        store: dict,
        lock: threading.Lock,
    ) -> EngineWorker:
        key = (model_path, backend.name)
        with lock:
            if key not in store:
                store[key] = EngineWorker(self.lib_path, backend, model_path=model_path)
            return store[key]

    def _cpu_infer(self, prompt: str, max_tokens: int, model_path: str | None) -> InferResult:
        worker = self._get_worker(model_path, BackendKind.CPU, self._cpu_workers, self._cpu_lock)
        text = worker.infer(prompt, max_tokens)
        return InferResult(text=text, device="cpu", warnings=[], model=model_path, format="nanoq")

    def _gpu_infer(self, prompt: str, max_tokens: int, model_path: str | None) -> InferResult:
        if not self._gpu_state.backend:
            raise RuntimeError("GPU backend unavailable")
        worker = self._get_worker(
            model_path, self._gpu_state.backend, self._gpu_workers_map, self._gpu_lock,
        )
        text = worker.infer(prompt, max_tokens)
        return InferResult(
            text=text,
            device=self._gpu_state.backend_name,
            warnings=[],
            model=model_path,
            format="nanoq",
        )

    def _resolve_device(self, device: DeviceLiteral) -> tuple[str, list[str]]:
        if device == "cpu":
            return "cpu", []
        if device == "gpu":
            if self._gpu_state.available:
                return self._gpu_state.backend_name, []
            return "cpu", ["GPU requested but unavailable; fell back to CPU"]
        if self._gpu_state.available:
            return self._gpu_state.backend_name, []
        logger.debug("auto device: no GPU backend, using CPU")
        return "cpu", []

    def submit(
        self,
        prompt: str,
        max_tokens: int,
        device: DeviceLiteral = "cpu",
        model_path: str | None = None,
    ):
        target, warnings = self._resolve_device(device)
        use_gpu = target in ("cuda", "opencl")

        if use_gpu and self.gpu_executor:
            fut = self.gpu_executor.submit(self._gpu_infer, prompt, max_tokens, model_path)

            def wrapped():
                try:
                    result = fut.result()
                    if warnings:
                        result.warnings.extend(warnings)
                    return result
                except Exception:
                    cpu_result = self._cpu_infer(prompt, max_tokens, model_path)
                    cpu_result.warnings.append("GPU execution failed; fell back to CPU")
                    cpu_result.warnings.extend(warnings)
                    return cpu_result

            return self.cpu_executor.submit(wrapped)

        def cpu_wrapped():
            result = self._cpu_infer(prompt, max_tokens, model_path)
            result.warnings.extend(warnings)
            return result

        return self.cpu_executor.submit(cpu_wrapped)
