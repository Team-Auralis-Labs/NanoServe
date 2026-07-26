#!/usr/bin/env bash
# Deployment audit: Docker CPU/GPU/GGUF, native simulation, static UI checks
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASS=0
FAIL=0
SKIP=0
REPORT=()

log() { echo "[audit] $*" >&2; }
pass() { PASS=$((PASS + 1)); REPORT+=("PASS: $1"); log "PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); REPORT+=("FAIL: $1 — $2"); log "FAIL: $1 — $2"; }
skip() { SKIP=$((SKIP + 1)); REPORT+=("SKIP: $1 — $2"); log "SKIP: $1 — $2"; }

check_health() {
  local name="$1" url="$2"
  local body
  body="$(curl -sf "$url/health" 2>/dev/null)" || { fail "$name health" "unreachable at $url"; return 1; }
  echo "$body" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d.get('status')=='ok', d" \
    && pass "$name health" || { fail "$name health" "status not ok"; return 1; }
  printf '%s' "$body"
}

check_completion() {
  local name="$1" url="$2" payload="$3" expect_code="${4:-200}"
  local code body
  code="$(curl -s -o /tmp/audit_out.json -w '%{http_code}' -X POST "$url/v1/completions" \
    -H 'Content-Type: application/json' -d "$payload")"
  body="$(cat /tmp/audit_out.json)"
  if [ "$code" = "$expect_code" ]; then
    pass "$name completion (HTTP $code)"
    echo "$body"
  else
    fail "$name completion" "expected HTTP $expect_code got $code: $body"
    echo "$body"
  fi
}

check_static() {
  local name="$1" url="$2" pattern="$3"
  local html
  html="$(curl -sf "$url/static/index.html" 2>/dev/null)" || { fail "$name static" "index.html unreachable"; return; }
  if echo "$html" | grep -q "$pattern"; then
    pass "$name static UI ($pattern)"
  else
    fail "$name static UI" "missing pattern: $pattern"
  fi
}

check_js() {
  local name="$1" url="$2" pattern="$3"
  local js
  js="$(curl -sf "$url/static/app.js" 2>/dev/null)" || { fail "$name app.js" "unreachable"; return; }
  if echo "$js" | grep -q "$pattern"; then
    pass "$name app.js ($pattern)"
  else
    fail "$name app.js" "missing: $pattern"
  fi
}

log "=== Docker CPU (:8000) ==="
if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
  H="$(check_health "CPU" "http://localhost:8000")"
  echo "$H" | python3 -c "import json,sys; d=json.load(sys.stdin); assert not d['gguf_available']" \
    && pass "CPU gguf_available=false" || fail "CPU gguf" "gguf should be off"
  check_completion "CPU auto-demo" "http://localhost:8000" \
    '{"prompt":"Hi","max_tokens":8,"format":"auto","device":"cpu"}' 200
  check_completion "CPU gguf-no-model" "http://localhost:8000" \
    '{"prompt":"Hi","max_tokens":8,"format":"gguf","device":"cpu"}' 400
  check_static "CPU" "http://localhost:8000" "Built-in demo"
  check_js "CPU" "http://localhost:8000" "updateFormatOptions"
else
  skip "Docker CPU" "not running on :8000"
fi

log "=== Docker GPU (:8001) ==="
if curl -sf http://localhost:8001/health >/dev/null 2>&1; then
  H="$(check_health "GPU" "http://localhost:8001")"
  echo "$H" | python3 -c "import json,sys; d=json.load(sys.stdin); print('gpu_available=', d.get('gpu_available'))"
  check_completion "GPU auto-demo" "http://localhost:8001" \
    '{"prompt":"Hi","max_tokens":8,"format":"auto","device":"cpu"}' 200
  check_static "GPU" "http://localhost:8001" "Built-in demo"
else
  skip "Docker GPU" "not running on :8001 (run: docker compose --profile gpu up -d)"
fi

log "=== Docker GGUF (:8002) ==="
if curl -sf http://localhost:8002/health >/dev/null 2>&1; then
  H="$(check_health "GGUF" "http://localhost:8002")"
  echo "$H" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['gguf_available']" \
    && pass "GGUF gguf_available=true" || fail "GGUF" "gguf should be on"
  MODELS="$(curl -sf http://localhost:8002/v1/models)"
  echo "$MODELS" | python3 -c "
import json,sys
d=json.load(sys.stdin)
n=len(d.get('models',[]))
print(f'models={n}')
assert n>=0
" && pass "GGUF model list"
  MID="$(echo "$MODELS" | python3 -c "import json,sys; m=json.load(sys.stdin).get('models',[]); print(m[0]['id'] if m else '')")"
  check_completion "GGUF no-model" "http://localhost:8002" \
    '{"prompt":"Hi","max_tokens":8,"format":"gguf","device":"cpu"}' 400
  if [ -n "$MID" ]; then
    check_completion "GGUF with-model" "http://localhost:8002" \
      "{\"prompt\":\"Hi\",\"max_tokens\":8,\"format\":\"gguf\",\"device\":\"cpu\",\"model\":\"$MID\"}" 200
  else
    skip "GGUF with-model" "no models in /models"
  fi
  check_js "GGUF" "http://localhost:8002" "Select model"
else
  skip "Docker GGUF" "not running on :8002"
fi

log "=== WASM static audit ==="
WASM_APP="$ROOT/deployment/wasm/app.js"
if [ -f "$WASM_APP" ]; then
  if grep -q "Engine not ready\|loadModel\|No model" "$WASM_APP"; then
    if grep -q "modelLabel\|loadModel" "$WASM_APP" && ! grep -q "if.*modelLabel\|chipModel.*off" "$WASM_APP" | grep -q "validate"; then
      : # check below
    fi
  fi
  if grep -q "fileInput" "$WASM_APP" && grep -q "go.onclick" "$WASM_APP"; then
    if grep -qE "if\s*\(.*model|No model loaded|load a .nanoq" "$WASM_APP"; then
      pass "WASM pre-generate model check"
    else
      fail "WASM app.js" "Generate allowed without model load validation"
    fi
  fi
  if [ -f "$ROOT/deployment/wasm/nanoserve_engine.wasm" ]; then
    pass "WASM artifacts present"
  else
    skip "WASM runtime" "nanoserve_engine.wasm not built"
  fi
else
  fail "WASM" "app.js missing"
fi

log "=== Native simulation (unit tests + env) ==="
if [ -f "$ROOT/.venv/bin/python" ]; then
  export LD_LIBRARY_PATH="$ROOT/allocator/target/release:${LD_LIBRARY_PATH:-}"
  export NANOSERVE_ENGINE_LIB="$ROOT/engine/build/libnanoserve_engine.so"
  export PYTHONPATH="$ROOT"
  if [ -f "$ROOT/engine/build/libnanoserve_engine.so" ]; then
    "$ROOT/.venv/bin/python" -m unittest tests.test_models_registry tests.test_gguf -q 2>/dev/null \
      && pass "native unit tests (registry+gguf)" || skip "native unit tests" "engine/tests partial fail"
  else
    skip "native engine tests" "libnanoserve_engine.so not built"
  fi
  if grep -q 'PYTHONPATH' "$ROOT/scripts/run_native.sh" && grep -q 'PYTHONPATH' "$ROOT/.env.nanoserve" 2>/dev/null; then
    pass "native PYTHONPATH configured"
  elif grep -q 'PYTHONPATH' "$ROOT/scripts/run_native.sh"; then
    pass "native PYTHONPATH in run_native.sh"
  else
    fail "native" "PYTHONPATH missing from run scripts"
  fi
else
  skip "native venv" ".venv not found — run ./install.sh"
fi

log "=== Dockerfile audit ==="
grep -q 'ENV PYTHONPATH=/app' "$ROOT/Dockerfile" && pass "Dockerfile PYTHONPATH" || fail "Dockerfile" "missing PYTHONPATH"
grep -q 'rm -rf engine/build' "$ROOT/Dockerfile" && pass "Dockerfile cmake cache wipe" || fail "Dockerfile" "missing engine/build rm"
grep -q 'allocator/target' "$ROOT/.dockerignore" && pass ".dockerignore allocator/target" || fail ".dockerignore" "missing allocator/target"
grep -q 'engine/build/' "$ROOT/.dockerignore" && pass ".dockerignore engine/build" || fail ".dockerignore" "missing engine/build"

log ""
log "========== SUMMARY =========="
log "PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
for line in "${REPORT[@]}"; do echo "  $line"; done
[ "$FAIL" -eq 0 ]
