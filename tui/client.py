"""Terminal chat client for NanoServe (uses `rich` for nice output)."""
import argparse
import sys

import httpx
from rich.console import Console
from rich.prompt import Prompt

console = Console()

HELP_TEXT = """
Slash commands:
  /device cpu|gpu|auto  — switch compute engine
  /help                 — show this help
  exit / quit           — leave the REPL
"""


def main():
    ap = argparse.ArgumentParser(description="NanoServe TUI client")
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:8000")
    ap.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    args = ap.parse_args()

    current_device = args.device
    console.print(f"[bold cyan]NanoServe TUI[/] connected to {args.url}")
    console.print(f"[dim]device={current_device} · type /help for commands · exit to quit[/]\n")

    with httpx.Client(base_url=args.url, timeout=30.0) as client:
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
            if lower == "/help":
                console.print(HELP_TEXT)
                continue

            r = client.post(
                "/v1/completions",
                json={"prompt": prompt, "max_tokens": 32, "device": current_device},
            )
            r.raise_for_status()
            data = r.json()
            console.print(f"[bold magenta]nano[/] {data['text']}")
            meta = f"device={data.get('device', 'cpu')} · {data['latency_ms']:.1f} ms"
            if data.get("warnings"):
                meta += " · " + "; ".join(data["warnings"])
            console.print(f"[dim]({meta})[/]\n")


if __name__ == "__main__":
    main()
