#!/usr/bin/env bash
# Build browser WASM demo (opt-in fourth tier).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/deployment/wasm/build.sh" "$@"
