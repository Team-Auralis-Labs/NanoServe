#!/usr/bin/env python3
"""Production-style load test: N concurrent users against /v1/completions."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass
class LoadResult:
    users: int
    device: str
    success: int = 0
    errors: list[str] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    wall_s: float = 0.0

    def to_dict(self) -> dict:
        lat = sorted(self.latencies_ms)
        n = len(lat)
        return {
            "users": self.users,
            "device": self.device,
            "success": self.success,
            "failed": len(self.errors),
            "success_rate_pct": round(100.0 * self.success / self.users, 2) if self.users else 0,
            "wall_clock_s": round(self.wall_s, 3),
            "throughput_rps": round(self.success / self.wall_s, 2) if self.wall_s > 0 else 0,
            "latency_ms": {
                "mean": round(statistics.mean(lat), 2) if lat else None,
                "p50": round(lat[n // 2], 2) if lat else None,
                "p95": round(lat[int(n * 0.95) - 1], 2) if n >= 20 else (round(lat[-1], 2) if lat else None),
                "p99": round(lat[int(n * 0.99) - 1], 2) if n >= 100 else (round(lat[-1], 2) if lat else None),
                "max": round(max(lat), 2) if lat else None,
                "min": round(min(lat), 2) if lat else None,
            },
            "errors_sample": self.errors[:5],
        }


def one_request(base_url: str, i: int, device: str, max_tokens: int) -> tuple[bool, float, str | None]:
    t0 = time.perf_counter()
    try:
        r = httpx.post(
            f"{base_url}/v1/completions",
            json={"prompt": f"user {i} concurrent production test", "max_tokens": max_tokens, "device": device},
            timeout=60.0,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("text"):
            return False, 0.0, "empty response"
        return True, (time.perf_counter() - t0) * 1000, None
    except Exception as e:
        return False, 0.0, str(e)


def run_load(base_url: str, users: int, device: str, max_tokens: int) -> LoadResult:
    result = LoadResult(users=users, device=device)
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=users) as ex:
        futures = [ex.submit(one_request, base_url, i, device, max_tokens) for i in range(users)]
        for f in as_completed(futures):
            ok, lat, err = f.result()
            if ok:
                result.success += 1
                result.latencies_ms.append(lat)
            elif err:
                result.errors.append(err)
    result.wall_s = time.perf_counter() - t_start
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="NanoServe multi-user load test")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--users", type=int, default=50)
    ap.add_argument("--preset", choices=["50", "150", "300"], default=None,
                    help="Shortcut: 50, 150, or 300 concurrent users")
    ap.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=1, help="Repeat load rounds back-to-back")
    ap.add_argument("--out", type=str, default="", help="Write JSON report to file")
    args = ap.parse_args()
    if args.preset:
        args.users = int(args.preset)

    all_rounds = []
    for r in range(args.rounds):
        print(f"[*] Round {r + 1}/{args.rounds}: {args.users} concurrent users (device={args.device}) ...")
        res = run_load(args.url, args.users, args.device, args.max_tokens)
        d = res.to_dict()
        all_rounds.append(d)
        print(f"    success={d['success']}/{args.users}  throughput={d['throughput_rps']} req/s")
        if d["latency_ms"]["p50"]:
            print(f"    p50={d['latency_ms']['p50']}ms  p95={d['latency_ms']['p95']}ms  max={d['latency_ms']['max']}ms")
        if d["failed"]:
            print(f"    errors={d['failed']}  sample={d['errors_sample']}")

    report = {"rounds": all_rounds, "passed": all(r["failed"] == 0 and r["success"] == args.users for r in all_rounds)}
    print("\n=== LOAD TEST REPORT ===")
    print(json.dumps(report, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"[+] wrote {args.out}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
