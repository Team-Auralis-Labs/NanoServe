"""NanoServe: a tiny vLLM-style orchestrator.

Async FastAPI + micro-batching + bounded engine thread pool.
Tuned for 150–300 concurrent users via server/config.py env vars.
"""
import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nanoserve.engine.pool import EnginePool

from server.config import BATCH_WINDOW_S, MAX_BATCH, MAX_QUEUE, NUM_WORKERS, UVICORN_WORKERS

APP_DIR = Path(__file__).parent

app = FastAPI(title="NanoServe")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

pool = EnginePool(num_workers=NUM_WORKERS)
queue: Optional[asyncio.Queue] = None
_queue_depth = 0
_queue_lock: Optional[asyncio.Lock] = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 24
    device: Literal["cpu", "gpu", "auto"] = "cpu"


class GenerateResponse(BaseModel):
    id: str
    text: str
    latency_ms: float
    device: str = "cpu"
    warnings: list[str] = Field(default_factory=list)


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

        for req_id, prompt, max_tokens, device, fut, t0 in batch:
            engine_future = pool.submit(prompt, max_tokens, device=device)
            asyncio.ensure_future(_resolve(engine_future, fut, t0))


async def _resolve(engine_future, fut: asyncio.Future, t0: float):
    global _queue_depth
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, engine_future.result)
        if not fut.done():
            fut.set_result((result.text, result.device, result.warnings, (time.perf_counter() - t0) * 1000))
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


@app.get("/", response_class=HTMLResponse)
async def index():
    return (APP_DIR / "static" / "index.html").read_text()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "workers": NUM_WORKERS,
        "max_batch": MAX_BATCH,
        "max_queue": MAX_QUEUE,
        "queue_depth": _queue_depth,
        "uvicorn_workers": UVICORN_WORKERS,
        "gpu_cuda": pool.gpu_cuda_available,
        "gpu_opencl": pool.gpu_opencl_available,
        "gpu_available": pool.gpu_available,
    }


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
    await queue.put((req_id, req.prompt, req.max_tokens, req.device, fut, time.perf_counter()))
    text, device, warnings, latency_ms = await fut
    return GenerateResponse(
        id=req_id,
        text=text,
        latency_ms=latency_ms,
        device=device,
        warnings=warnings,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
