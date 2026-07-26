"""NanoServe: a tiny vLLM-style orchestrator.

Async FastAPI + micro-batching + bounded engine thread pool.
Tuned for 150–300 concurrent users via server/config.py env vars.
"""
from __future__ import annotations
import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nanoserve.engine.gguf_probe import gguf_available
from nanoserve.engine.gguf_worker import gguf_model_loaded
from nanoserve.engine.router import InferenceRouter
from nanoserve.models.download import download_model
from nanoserve.models.pipeline import _auto_quantize_default
from nanoserve.models.registry import ModelEntry, ModelRegistry

from server.config import BATCH_WINDOW_S, MAX_BATCH, MAX_QUEUE, NUM_WORKERS, UVICORN_WORKERS

APP_DIR = Path(__file__).parent

app = FastAPI(title="NanoServe")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

registry = ModelRegistry()
router = InferenceRouter(num_workers=NUM_WORKERS, registry=registry)
queue: Optional[asyncio.Queue] = None
_queue_depth = 0
_queue_lock: Optional[asyncio.Lock] = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 24
    device: Literal["cpu", "gpu", "auto"] = "cpu"
    model: Optional[str] = None
    format: Literal["auto", "nanoq", "gguf"] = "auto"
    quantize: Optional[bool] = None
    precision: Literal["int8", "fp16", "fp4", "raw"] = "int8"


class GenerateResponse(BaseModel):
    id: str
    text: str
    latency_ms: float
    device: str = "cpu"
    format: str = "nanoq"
    model: Optional[str] = None
    quantized: bool = True
    warnings: list[str] = Field(default_factory=list)


class DownloadRequest(BaseModel):
    source: Literal["hf", "url"]
    model_id: Optional[str] = None
    repo_id: Optional[str] = None
    url: Optional[str] = None
    filename: Optional[str] = None
    revision: Optional[str] = None
    quantize: Optional[bool] = None
    precision: Literal["int8", "fp16", "fp4", "raw"] = "int8"


class ModelResponse(BaseModel):
    id: str
    source_path: str
    nanoq_path: Optional[str] = None
    format: str
    dtype: str
    rows: int
    cols: int
    size_bytes: int
    quantized: bool
    source: str
    loaded: bool = False


async def batcher_loop():
    loop = asyncio.get_running_loop()
    assert queue is not None
    while True:
        batch = [await queue.get()]
        deadline = loop.time() + BATCH_WINDOW_S
        while len(batch) < MAX_BATCH:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout=timeout))
            except asyncio.TimeoutError:
                break

        for item in batch:
            (
                req_id, prompt, max_tokens, device, model, fmt,
                quantize, precision, fut, t0,
            ) = item
            engine_future = router.submit(
                prompt,
                max_tokens,
                device=device,
                model=model,
                format=fmt,
                quantize=quantize,
                precision=precision,
            )
            asyncio.ensure_future(_resolve(engine_future, fut, t0))


async def _resolve(engine_future, fut: asyncio.Future, t0: float):
    global _queue_depth
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, engine_future.result)
        if not fut.done():
            fut.set_result((
                result.text,
                result.device,
                result.format,
                result.model,
                result.quantized,
                result.warnings,
                (time.perf_counter() - t0) * 1000,
            ))
    finally:
        if _queue_lock:
            async with _queue_lock:
                _queue_depth = max(0, _queue_depth - 1)


@app.on_event("startup")
async def startup():
    global queue, _queue_lock
    queue = asyncio.Queue()
    _queue_lock = asyncio.Lock()
    asyncio.create_task(batcher_loop())


@app.get("/")
async def index():
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "workers": NUM_WORKERS,
        "max_batch": MAX_BATCH,
        "max_queue": MAX_QUEUE,
        "queue_depth": _queue_depth,
        "uvicorn_workers": UVICORN_WORKERS,
        "gpu_cuda": router.gpu_cuda_available,
        "gpu_opencl": router.gpu_opencl_available,
        "gpu_available": router.gpu_available,
        "native_available": True,
        "gguf_available": gguf_available(),
        "gguf_model_loaded": gguf_model_loaded(),
        "active_format": router.active_format(),
        "default_format": os.environ.get("NANOSERVE_DEFAULT_FORMAT", "auto"),
        "models_registered": registry.count,
        "models_loaded": router.model_cache.loaded_count(),
        "auto_quantize": _auto_quantize_default(),
    }


def _entry_response(entry: ModelEntry) -> ModelResponse:
    loaded = False
    path = entry.nanoq_path or entry.source_path
    if path:
        loaded = router.model_cache.is_loaded(path, "cpu") or router.model_cache.is_loaded(path, "gpu")
    return ModelResponse(
        id=entry.id,
        source_path=entry.source_path,
        nanoq_path=entry.nanoq_path,
        format=entry.format,
        dtype=entry.dtype,
        rows=entry.rows,
        cols=entry.cols,
        size_bytes=entry.size_bytes,
        quantized=entry.quantized,
        source=entry.source,
        loaded=loaded,
    )


@app.get("/v1/models")
async def list_models():
    return {"models": [_entry_response(e).model_dump() for e in registry.list()]}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    entry = registry.get(model_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Model not found")
    return _entry_response(entry)


@app.post("/v1/models/download")
async def download(req: DownloadRequest):
    try:
        entry = download_model(
            source=req.source,
            model_id=req.model_id,
            repo_id=req.repo_id,
            url=req.url,
            filename=req.filename,
            revision=req.revision,
            registry=registry,
        )
        from nanoserve.models.pipeline import prepare_model

        path, fmt, updated, warnings, quantized = prepare_model(
            entry.id,
            registry=registry,
            quantize=req.quantize,
            precision=req.precision,
        )
        if updated:
            entry = updated
        return {
            "model": _entry_response(entry).model_dump(),
            "resolved_path": path,
            "format": fmt,
            "quantized": quantized,
            "warnings": warnings,
        }
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/v1/models/{model_id}")
async def delete_model(model_id: str):
    if not registry.delete(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    return {"deleted": model_id}


@app.post("/v1/completions", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    global _queue_depth
    assert queue is not None and _queue_lock is not None

    async with _queue_lock:
        if MAX_QUEUE > 0 and _queue_depth >= MAX_QUEUE:
            raise HTTPException(status_code=503, detail="Server busy; retry later")
        _queue_depth += 1

    req_id = str(uuid.uuid4())
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    await queue.put((
        req_id,
        req.prompt,
        req.max_tokens,
        req.device,
        req.model,
        req.format,
        req.quantize,
        req.precision,
        fut,
        time.perf_counter(),
    ))
    text, device, fmt, model, quantized, warnings, latency_ms = await fut
    return GenerateResponse(
        id=req_id,
        text=text,
        latency_ms=latency_ms,
        device=device,
        format=fmt,
        model=model,
        quantized=quantized,
        warnings=warnings,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
