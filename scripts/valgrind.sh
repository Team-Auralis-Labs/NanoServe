#!/usr/bin/env bash
# Valgrind on pure C engine loop (no Python interpreter noise)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LIB="$ROOT/engine/build/libnanoserve_engine.so"
ALLOC="$ROOT/allocator/target/release"

if [ ! -f "$LIB" ]; then
  echo "[!] Build engine first" >&2
  exit 1
fi

if ! command -v valgrind >/dev/null 2>&1; then
  echo "[!] sudo apt-get install -y valgrind" >&2
  exit 1
fi

CFLAGS=(-std=c17 -O0 -g)
LDFLAGS=(-L"$ALLOC" -L"$ROOT/engine/build" -lnanoserve_engine -lbuddy_alloc -lpthread -ldl)
export LD_LIBRARY_PATH="$ALLOC:$ROOT/engine/build"

run_valgrind() {
  local bin=$1 log=$2 label=$3
  echo "[*] Valgrind $label ..."
  valgrind --leak-check=full --show-leak-kinds=all \
    --errors-for-leak-kinds=definite,indirect \
    --error-exitcode=42 \
    "$bin" 2>&1 | tee "$log"
}

BIN="$ROOT/tests/valgrind_engine"
gcc "${CFLAGS[@]}" -o "$BIN" "$ROOT/tests/valgrind_engine.c" "${LDFLAGS[@]}"
LOG="$ROOT/documentation/valgrind_report.txt"
run_valgrind "$BIN" "$LOG" "C engine harness (200 cycles)"

CYCLES="${VALGRIND_CYCLES:-1000}"
BIN_EXT="$ROOT/tests/valgrind_engine_ext"
gcc "${CFLAGS[@]}" -o "$BIN_EXT" "$ROOT/tests/valgrind_engine_ext.c" "${LDFLAGS[@]}"
LOG_EXT="$ROOT/documentation/valgrind_report_extended.txt"
export VALGRIND_CYCLES="$CYCLES"
run_valgrind "$BIN_EXT" "$LOG_EXT" "extended ($CYCLES cycles)"

echo "[+] $LOG"
echo "[+] $LOG_EXT"
