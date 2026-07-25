#!/usr/bin/env python3
"""Monitor server RSS during HTTP load; detect runaway growth after warmup."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return -1.0
    return -1.0


def post(url: str, i: int) -> bool:
    body = json.dumps({"prompt": f"rss audit {i}", "max_tokens": 16, "device": "cpu"}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status == 200
    except urllib.error.URLError:
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--pid", type=int, required=True, help="uvicorn/server PID")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--users", type=int, default=30)
    args = p.parse_args()

    samples: list[dict] = []
    req_id = 0

    for rnd in range(args.rounds):
        n = args.warmup if rnd == 0 else args.users
        print(f"[*] Round {rnd + 1}/{args.rounds}: {n} concurrent requests")
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(post, args.url, req_id + i) for i in range(n)]
            ok = sum(1 for f in as_completed(futs) if f.result())
        req_id += n
        elapsed = time.perf_counter() - t0
        r = rss_mb(args.pid)
        samples.append({"round": rnd + 1, "requests": n, "ok": ok, "rss_mb": r, "elapsed_s": round(elapsed, 2)})
        print(f"    ok={ok}/{n} RSS={r:.1f} MiB elapsed={elapsed:.1f}s")
        time.sleep(1)

    if len(samples) < 2:
        return 0

    baseline = samples[0]["rss_mb"]
    final = samples[-1]["rss_mb"]
    growth = final - baseline
    print(f"[*] RSS baseline (after warmup round): {baseline:.1f} MiB")
    print(f"[*] RSS final: {final:.1f} MiB (delta {growth:+.1f} MiB)")

    out = Path(__file__).resolve().parents[1] / "documentation" / "memory_rss_server.json"
    out.write_text(json.dumps({"samples": samples, "growth_mb": growth}, indent=2))
    print(f"[+] wrote {out}")

    if growth > 64:
        print(f"[!] WARN: server RSS grew {growth:.1f} MiB after warmup — investigate", file=sys.stderr)
        return 1
    print("[+] PASS: server RSS stable after warmup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
