"""Concurrent load tester: simulates N users hitting the server at once,
so you can prove it stays non-blocking and check p50/p95 latency."""
import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


def one_request(base_url: str, i: int, device: str):
    t0 = time.perf_counter()
    r = httpx.post(
        f"{base_url}/v1/completions",
        json={"prompt": f"user {i} says hello", "max_tokens": 16, "device": device},
        timeout=30.0,
    )
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--users", type=int, default=50)
    ap.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    args = ap.parse_args()

    print(f"[*] Firing {args.users} concurrent requests at {args.url} (device={args.device}) ...")
    latencies = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.users) as ex:
        futures = [ex.submit(one_request, args.url, i, args.device) for i in range(args.users)]
        for f in as_completed(futures):
            latencies.append(f.result())
    total = time.perf_counter() - t_start

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    print(f"[+] {args.users} users served in {total:.2f}s wall clock")
    print(f"    mean={statistics.mean(latencies):.1f}ms  p50={p50:.1f}ms  p95={p95:.1f}ms")


if __name__ == "__main__":
    main()
