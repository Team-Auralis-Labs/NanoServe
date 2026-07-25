"""Server tuning via environment (150–300+ concurrent users)."""
from __future__ import annotations

import os

# Engine thread pool — default: all logical CPU cores
NUM_WORKERS = int(os.environ.get("NANOSERVE_NUM_WORKERS", os.cpu_count() or 4))

# Micro-batcher: widen window + batch for high concurrency
BATCH_WINDOW_S = float(os.environ.get("NANOSERVE_BATCH_WINDOW_S", "0.025"))
MAX_BATCH = int(os.environ.get("NANOSERVE_MAX_BATCH", "32"))

# Max queued requests before 503 (0 = unlimited)
MAX_QUEUE = int(os.environ.get("NANOSERVE_MAX_QUEUE", "512"))

# Uvicorn/gunicorn process hint (documentation / health)
UVICORN_WORKERS = int(os.environ.get("NANOSERVE_UVICORN_WORKERS", "1"))
