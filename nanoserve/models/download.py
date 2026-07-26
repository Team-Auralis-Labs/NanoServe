"""Download models from HuggingFace Hub or direct URL."""
from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from nanoserve.models.registry import ModelEntry, ModelRegistry, _models_dir


def _safe_id(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-") or "model"


def download_model(
    *,
    source: str,
    model_id: str | None = None,
    repo_id: str | None = None,
    url: str | None = None,
    filename: str | None = None,
    revision: str | None = None,
    registry: ModelRegistry | None = None,
) -> ModelEntry:
    reg = registry or ModelRegistry()
    dest_root = reg.root

    if source == "hf":
        if not repo_id:
            raise ValueError("repo_id required for HuggingFace download")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface-hub required: pip install nanoserve[models]") from exc

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename or "model.safetensors",
            revision=revision,
            local_dir=str(dest_root / _safe_id(model_id or repo_id.replace("/", "--"))),
        )
        resolved_id = _safe_id(model_id or repo_id.replace("/", "--"))
        entry = ModelEntry(
            id=resolved_id,
            source_path=str(Path(local_path).resolve()),
            format="safetensors" if str(local_path).endswith(".safetensors") else "bin",
            source="hf",
            extra={"repo_id": repo_id, "filename": filename, "revision": revision},
        )
    elif source == "url":
        if not url:
            raise ValueError("url required for URL download")
        resolved_id = _safe_id(model_id or Path(url).name)
        dest_dir = dest_root / resolved_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        fname = filename or Path(url.split("?")[0]).name or "model.bin"
        dest = dest_dir / fname
        urllib.request.urlretrieve(url, dest)
        entry = ModelEntry(
            id=resolved_id,
            source_path=str(dest.resolve()),
            format=dest.suffix.lstrip(".") or "bin",
            source="url",
            extra={"url": url},
        )
    else:
        raise ValueError(f"unsupported source: {source}")

    entry.size_bytes = Path(entry.source_path).stat().st_size
    reg.register(entry)
    return entry
