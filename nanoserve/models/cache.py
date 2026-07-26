"""LRU cache for loaded native model paths per device."""
from __future__ import annotations

import os
import threading
from collections import OrderedDict


class ModelCache:
    def __init__(self, max_size: int | None = None):
        self.max_size = max_size or int(os.environ.get("NANOSERVE_MAX_LOADED_MODELS", "2"))
        self._lock = threading.Lock()
        self._keys: OrderedDict[tuple[str, str], str] = OrderedDict()

    def touch(self, model_path: str, device: str) -> None:
        key = (model_path, device)
        with self._lock:
            if key in self._keys:
                self._keys.move_to_end(key)
            else:
                self._keys[key] = model_path
                while len(self._keys) > self.max_size:
                    self._keys.popitem(last=False)

    def loaded_paths(self) -> list[str]:
        with self._lock:
            return list({v for v in self._keys.values()})

    def loaded_count(self) -> int:
        with self._lock:
            return len(self._keys)

    def is_loaded(self, model_path: str, device: str) -> bool:
        with self._lock:
            return (model_path, device) in self._keys

    def clear(self) -> None:
        with self._lock:
            self._keys.clear()
