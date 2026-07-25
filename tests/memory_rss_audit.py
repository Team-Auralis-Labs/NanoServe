#!/usr/bin/env python3
"""Memory audit: RSS plateau after warmup + repeated engine cycles.

Expected behavior (NOT a leak):
  - RSS rises while EnginePool lazily creates one engine per worker thread
    (each scratch_pool ~= 16 MiB → NUM_WORKERS * 16 MiB total).
  - RSS then stabilizes: buddy pools reuse memory; per-infer allocs are freed.

Exit 0 if RSS growth after warmup <= threshold_mb.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "NANOSERVE_ENGINE_LIB",
    str(ROOT / "engine" / "build" / "libnanoserve_engine.so"),
)
os.environ.setdefault(
    "LD_LIBRARY_PATH",
    f"{ROOT / 'allocator' / 'target' / 'release'}:{os.environ.get('LD_LIBRARY_PATH', '')}",
)

WARMUP = int(os.environ.get("MEM_AUDIT_WARMUP", "32"))
SUSTAINED = int(os.environ.get("MEM_AUDIT_SUSTAINED", "400"))
THRESHOLD_MB = float(os.environ.get("MEM_AUDIT_THRESHOLD_MB", "48"))


def rss_mb() -> float:
    """Current process RSS in MiB via /proc."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def main() -> int:
    lib = Path(os.environ["NANOSERVE_ENGINE_LIB"])
    if not lib.exists():
        print(f"[!] Engine not built: {lib}", file=sys.stderr)
        return 2

    from nanoserve.engine.pool import EnginePool

    num_workers = int(os.environ.get("NANOSERVE_NUM_WORKERS", "4"))
    pool = EnginePool(num_workers=num_workers)
    samples: list[tuple[int, float]] = []

    def one(i: int) -> None:
        fut = pool.submit(f"mem audit {i}", 16, device="cpu")
        fut.result()

    print(f"[*] Memory audit: workers={num_workers} warmup={WARMUP} sustained={SUSTAINED}")
    print(f"[*] RSS at start: {rss_mb():.1f} MiB")

    for i in range(WARMUP + SUSTAINED):
        one(i)
        if i == WARMUP - 1:
            baseline = rss_mb()
            print(f"[*] RSS after warmup ({WARMUP} inferences): {baseline:.1f} MiB")
        if i >= WARMUP and (i - WARMUP) % 50 == 0:
            r = rss_mb()
            samples.append((i, r))
            print(f"    step {i}: RSS {r:.1f} MiB")

    final = rss_mb()
    growth = final - baseline
    print(f"[*] RSS after sustained load: {final:.1f} MiB (delta since warmup: {growth:+.1f} MiB)")

    if growth > THRESHOLD_MB:
        print(
            f"[!] FAIL: RSS grew {growth:.1f} MiB after warmup (threshold {THRESHOLD_MB} MiB)",
            file=sys.stderr,
        )
        return 1

    print(f"[+] PASS: RSS stabilized (growth {growth:.1f} MiB <= {THRESHOLD_MB} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
