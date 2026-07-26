"""Model registry backed by ~/.nanoserve/models."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _models_dir() -> Path:
    return Path(os.environ.get("NANOSERVE_MODELS_DIR", Path.home() / ".nanoserve" / "models"))


@dataclass
class ModelEntry:
    id: str
    source_path: str
    nanoq_path: str | None = None
    format: str = "nanoq"
    dtype: str = "int8"
    rows: int = 0
    cols: int = 0
    size_bytes: int = 0
    quantized: bool = True
    source: str = "local"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    def __init__(self, root: Path | None = None):
        self.root = root or _models_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest = self.root / "registry.json"
        self._entries: dict[str, ModelEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self._manifest.exists():
            self._save()
            return
        data = json.loads(self._manifest.read_text())
        for item in data.get("models", []):
            entry = ModelEntry(**{k: v for k, v in item.items() if k in ModelEntry.__dataclass_fields__})
            self._entries[entry.id] = entry

    def _save(self) -> None:
        payload = {"models": [e.to_dict() for e in self._entries.values()]}
        self._manifest.write_text(json.dumps(payload, indent=2))

    def register(self, entry: ModelEntry) -> ModelEntry:
        model_dir = self.root / entry.id
        model_dir.mkdir(parents=True, exist_ok=True)
        self._entries[entry.id] = entry
        self._save()
        return entry

    def get(self, model_id: str) -> ModelEntry | None:
        return self._entries.get(model_id)

    def list(self) -> list[ModelEntry]:
        return list(self._entries.values())

    def resolve_path(self, model: str | None) -> str | None:
        if not model:
            env = os.environ.get("NANOSERVE_MODEL_PATH")
            return env or None
        if model in self._entries:
            e = self._entries[model]
            return e.nanoq_path or e.source_path
        p = Path(model)
        if p.exists():
            return str(p.resolve())
        candidate = self.root / model
        if candidate.exists():
            return str(candidate.resolve())
        nanoq = self.root / model / f"{model}.nanoq"
        if nanoq.exists():
            return str(nanoq.resolve())
        return model

    def delete(self, model_id: str) -> bool:
        if model_id not in self._entries:
            return False
        del self._entries[model_id]
        model_dir = self.root / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        self._save()
        return True

    @property
    def count(self) -> int:
        return len(self._entries)
