#!/usr/bin/env python3
"""RSS under concurrent load — shows warmup growth as worker threads spin up."""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NANOSERVE_ENGINE_LIB", str(ROOT / "engine/build/libnanoserve_engine.so"))
os.environ.setdefault(
    "LD_LIBRARY_PATH",
    f"{ROOT / 'allocator/target/release'}:{os.environ.get('LD_LIBRARY_PATH', '')}",
)


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def main() -> int:
    from nanoserve.engine.pool import EnginePool

    workers = int(os.environ.get("NANOSERVE_NUM_WORKERS", "8"))
    pool = EnginePool(num_workers=workers)
    print(f"[*] Concurrent RSS audit: {workers} pool workers")
    print(f"[*] Start RSS: {rss_mb():.1f} MiB (expect growth as threads init engines)")

    def job(i: int) -> int:
        return pool.submit(f"concurrent {i}", 16, device="cpu").result().text

    # Burst 1: all workers at once
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(job, range(workers)))
    print(f"[*] After 1st burst ({workers} parallel): RSS={rss_mb():.1f} MiB")

    # Sustained bursts
    for burst in range(1, 6):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(job, burst * 1000 + i) for i in range(workers * 2)]
            for _ in as_completed(futs):
                pass
        print(f"[*] After burst {burst + 1}: RSS={rss_mb():.1f} MiB")

    r0 = rss_mb()
    for burst in range(6, 16):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(job, burst * 1000 + i) for i in range(workers * 2)]
            for _ in as_completed(futs):
                pass
    r1 = rss_mb()
    print(f"[*] After 10 more bursts: RSS={r1:.1f} MiB (delta {r1 - r0:+.1f} MiB)")
    if r1 - r0 > 32:
        print("[!] FAIL: RSS still climbing after warmup")
        return 1
    print("[+] PASS: RSS plateau after concurrent warmup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
