"""Terminal chat client for NanoServe (uses `rich` for nice output)."""
import argparse
import sys

import httpx
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

console = Console()

HELP_TEXT = """
Slash commands:
  /device cpu|gpu|auto       — switch compute engine
  /model [id|path]           — select model (empty = default)
  /format auto|nanoq|gguf    — model runtime format
  /precision int8|fp16|fp4|raw — quantization precision
  /models                    — list registered models
  /download hf <repo> [file] — download from HuggingFace
  /download url <url>        — download from URL
  /help                      — show this help
  exit / quit                — leave the REPL
"""


def main():
    ap = argparse.ArgumentParser(description="NanoServe TUI client")
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:8000")
    ap.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    ap.add_argument("--model", default=None)
    ap.add_argument("--format", choices=["auto", "nanoq", "gguf"], default="auto")
    ap.add_argument("--precision", choices=["int8", "fp16", "fp4", "raw"], default="int8")
    args = ap.parse_args()

    current_device = args.device
    current_model = args.model
    current_format = args.format
    current_precision = args.precision

    console.print(f"[bold cyan]NanoServe TUI[/] connected to {args.url}")
    console.print(
        f"[dim]device={current_device} · format={current_format} · "
        f"precision={current_precision} · /help for commands[/]\n"
    )

    with httpx.Client(base_url=args.url, timeout=120.0) as client:
        while True:
            prompt = Prompt.ask("[bold green]you[/]")
            stripped = prompt.strip()
            lower = stripped.lower()

            if lower in ("exit", "quit"):
                break
            if lower.startswith("/device"):
                parts = stripped.split()
                if len(parts) == 2 and parts[1] in ("cpu", "gpu", "auto"):
                    current_device = parts[1]
                    console.print(f"[dim]device set to {current_device}[/]")
                else:
                    console.print("[red]usage: /device cpu|gpu|auto[/]")
                continue
            if lower.startswith("/model"):
                parts = stripped.split(maxsplit=1)
                current_model = parts[1].strip() if len(parts) > 1 else None
                console.print(f"[dim]model set to {current_model or 'default'}[/]")
                continue
            if lower.startswith("/format"):
                parts = stripped.split()
                if len(parts) == 2 and parts[1] in ("auto", "nanoq", "gguf"):
                    current_format = parts[1]
                    console.print(f"[dim]format set to {current_format}[/]")
                else:
                    console.print("[red]usage: /format auto|nanoq|gguf[/]")
                continue
            if lower.startswith("/precision"):
                parts = stripped.split()
                if len(parts) == 2 and parts[1] in ("int8", "fp16", "fp4", "raw"):
                    current_precision = parts[1]
                    console.print(f"[dim]precision set to {current_precision}[/]")
                else:
                    console.print("[red]usage: /precision int8|fp16|fp4|raw[/]")
                continue
            if lower == "/models":
                r = client.get("/v1/models")
                r.raise_for_status()
                models = r.json().get("models", [])
                if not models:
                    console.print("[dim]no models registered[/]")
                    continue
                table = Table(title="Models")
                table.add_column("ID")
                table.add_column("Format")
                table.add_column("Dtype")
                table.add_column("Quantized")
                for m in models:
                    table.add_row(m["id"], m["format"], m["dtype"], str(m["quantized"]))
                console.print(table)
                continue
            if lower.startswith("/download"):
                parts = stripped.split()
                if len(parts) < 3:
                    console.print("[red]usage: /download hf <repo> [file] | /download url <url>[/]")
                    continue
                src = parts[1]
                body = {"source": src, "precision": current_precision}
                if src == "hf":
                    body["repo_id"] = parts[2]
                    if len(parts) > 3:
                        body["filename"] = parts[3]
                elif src == "url":
                    body["url"] = parts[2]
                else:
                    console.print("[red]source must be hf or url[/]")
                    continue
                r = client.post("/v1/models/download", json=body)
                if r.status_code >= 400:
                    console.print(f"[red]{r.text}[/]")
                    continue
                data = r.json()
                current_model = data["model"]["id"]
                console.print(f"[dim]downloaded model {current_model}[/]")
                continue
            if lower == "/help":
                console.print(HELP_TEXT)
                continue

            payload = {
                "prompt": prompt,
                "max_tokens": 32,
                "device": current_device,
                "format": current_format,
                "precision": current_precision,
            }
            if current_model:
                payload["model"] = current_model
            if current_precision == "raw":
                payload["quantize"] = False

            r = client.post("/v1/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            console.print(f"[bold magenta]nano[/] {data['text']}")
            meta = (
                f"device={data.get('device', 'cpu')} · format={data.get('format', 'nanoq')} · "
                f"quantized={data.get('quantized', True)} · {data['latency_ms']:.1f} ms"
            )
            if data.get("model"):
                meta = f"model={data['model']} · " + meta
            if data.get("warnings"):
                meta += " · " + "; ".join(data["warnings"])
            console.print(f"[dim]({meta})[/]\n")


if __name__ == "__main__":
    main()
