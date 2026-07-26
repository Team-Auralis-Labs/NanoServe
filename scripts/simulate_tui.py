#!/usr/bin/env python3
"""Simulate TUI client flows against a running NanoServe HTTP server."""
from __future__ import annotations

import argparse
import sys

import httpx


def main() -> int:
    ap = argparse.ArgumentParser(description="TUI flow simulation")
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:8000")
    ap.add_argument("--expect-gguf", action="store_true")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    ok = True

    with httpx.Client(base_url=base, timeout=120.0) as client:
        health = client.get("/health").json()
        print(f"health: gguf_available={health.get('gguf_available')} models={health.get('models_registered')}")

        # /models (TUI /models)
        models = client.get("/v1/models").json().get("models", [])
        print(f"/models: {len(models)} registered")

        # auto demo without model (TUI default chat)
        r = client.post("/v1/completions", json={
            "prompt": "Hello from TUI sim",
            "max_tokens": 8,
            "device": "cpu",
            "format": "auto",
        })
        if r.status_code != 200:
            print(f"FAIL auto completion: {r.status_code} {r.text}")
            ok = False
        else:
            print(f"OK auto: format={r.json().get('format')} text={r.json().get('text')[:40]!r}")

        # gguf without model should 400 when gguf available
        r = client.post("/v1/completions", json={
            "prompt": "Hi",
            "max_tokens": 8,
            "device": "cpu",
            "format": "gguf",
        })
        if args.expect_gguf:
            if r.status_code != 400:
                print(f"FAIL gguf-no-model: expected 400 got {r.status_code}")
                ok = False
            else:
                print("OK gguf-no-model rejected")
            if models:
                mid = models[0]["id"]
                r = client.post("/v1/completions", json={
                    "prompt": "Hi",
                    "max_tokens": 8,
                    "device": "cpu",
                    "format": "gguf",
                    "model": mid,
                })
                if r.status_code != 200:
                    print(f"FAIL gguf with model {mid}: {r.status_code} {r.text}")
                    ok = False
                else:
                    print(f"OK gguf model={mid} format={r.json().get('format')}")
        else:
            if r.status_code == 400:
                print("OK gguf-no-model rejected on non-gguf server")
            elif r.status_code == 200 and r.json().get("format") == "nanoq":
                print("OK gguf fell back to native on CPU-only server")
            else:
                print(f"WARN gguf on CPU server: {r.status_code} {r.text[:120]}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
