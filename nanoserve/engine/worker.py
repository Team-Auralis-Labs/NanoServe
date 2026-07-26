"""ctypes bindings to libnanoserve_engine.so."""
from __future__ import annotations

import ctypes
import enum
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

_LIB_CACHE: dict[str, ctypes.CDLL] = {}


def _default_lib_path() -> str:
    env = os.environ.get("NANOSERVE_ENGINE_LIB")
    if env:
        return env
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "engine" / "build" / "libnanoserve_engine.so")


def _load_lib(path: str) -> ctypes.CDLL:
    if path not in _LIB_CACHE:
        _LIB_CACHE[path] = ctypes.CDLL(path)
    return _LIB_CACHE[path]


class BackendKind(enum.IntEnum):
    CPU = 0
    CUDA = 1
    OPENCL = 2


@dataclass
class InferResult:
    text: str
    device: str
    warnings: list[str] = field(default_factory=list)
    format: str = "nanoq"
    model: str | None = None
    quantized: bool = True


class EngineWorker:
    """One engine handle per thread (thread-local)."""

    _local = threading.local()

    def __init__(
        self,
        lib_path: str | None = None,
        backend: BackendKind = BackendKind.CPU,
        model_path: str | None = None,
    ):
        path = lib_path or _default_lib_path()
        self.lib = _load_lib(path)
        self.lib.engine_init.restype = ctypes.c_void_p
        self.lib.engine_init_backend.restype = ctypes.c_void_p
        self.lib.engine_init_backend.argtypes = [ctypes.c_int]
        self.lib.engine_init_with_model.restype = ctypes.c_void_p
        self.lib.engine_init_with_model.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.engine_reload_model.restype = ctypes.c_int
        self.lib.engine_reload_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.engine_model_info.restype = ctypes.c_char_p
        self.lib.engine_model_info.argtypes = [ctypes.c_void_p]
        self.lib.engine_infer.restype = ctypes.c_int
        self.lib.engine_infer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self.lib.engine_probe_cuda.restype = ctypes.c_int
        self.lib.engine_probe_opencl.restype = ctypes.c_int
        self.lib.engine_backend_name.restype = ctypes.c_char_p
        self.lib.engine_backend_name.argtypes = [ctypes.c_int]
        self.lib.engine_cleanup.argtypes = [ctypes.c_void_p]
        self.backend = backend
        self.model_path = model_path

    @classmethod
    def probe_cuda(cls, lib_path: str | None = None) -> bool:
        lib = _load_lib(lib_path or _default_lib_path())
        lib.engine_probe_cuda.restype = ctypes.c_int
        return bool(lib.engine_probe_cuda())

    @classmethod
    def probe_opencl(cls, lib_path: str | None = None) -> bool:
        lib = _load_lib(lib_path or _default_lib_path())
        lib.engine_probe_opencl.restype = ctypes.c_int
        return bool(lib.engine_probe_opencl())

    def _handle(self):
        key = f"handle_{self.backend.name}_{self.model_path or 'default'}"
        if not hasattr(self._local, key):
            if self.model_path:
                handle = self.lib.engine_init_with_model(
                    self.model_path.encode("utf-8"), int(self.backend)
                )
            elif self.backend == BackendKind.CPU:
                handle = self.lib.engine_init()
            else:
                handle = self.lib.engine_init_backend(int(self.backend))
            if not handle:
                raise RuntimeError(f"Failed to init backend {self.backend.name}")
            setattr(self._local, key, handle)
        return getattr(self._local, key)

    def reload_model(self, model_path: str) -> None:
        rc = self.lib.engine_reload_model(self._handle(), model_path.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"Failed to reload model: {model_path}")
        self.model_path = model_path

    def model_info(self) -> str:
        info = self.lib.engine_model_info(self._handle())
        return info.decode("utf-8") if info else "{}"

    def cleanup(self) -> None:
        key = f"handle_{self.backend.name}_{self.model_path or 'default'}"
        if hasattr(self._local, key):
            handle = getattr(self._local, key)
            if handle:
                self.lib.engine_cleanup(handle)
            delattr(self._local, key)

    def infer(self, prompt: str, max_tokens: int = 24) -> str:
        buf = ctypes.create_string_buffer(4096)
        self.lib.engine_infer(
            self._handle(),
            prompt.encode("utf-8"),
            max_tokens,
            buf,
            len(buf),
        )
        return buf.value.decode("utf-8", errors="ignore")

    def backend_name(self) -> str:
        name = self.lib.engine_backend_name(int(self.backend))
        return name.decode("utf-8") if name else self.backend.name.lower()
