/**
 * NanoServe browser WASM API — thin wrapper over Emscripten Module.
 */
const NanoServeWasm = (() => {
  let Module = null;
  let handle = 0;
  let initFn = null;
  let initBytesFn = null;
  let reloadBytesFn = null;
  let inferFn = null;
  let modelInfoFn = null;
  let cleanupFn = null;

  const MAX_MODEL_BYTES = 16 * 1024 * 1024;
  const OUT_BUF = 4096;

  async function init(opts = {}) {
    const wasmJs = opts.scriptUrl || './nanoserve_engine.js';
    if (typeof createNanoServeModule === 'undefined') {
      await new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = wasmJs;
        s.onload = resolve;
        s.onerror = () => reject(new Error(`Failed to load ${wasmJs}`));
        document.head.appendChild(s);
      });
    }
    Module = await createNanoServeModule(
      opts.moduleConfig || { locateFile: (p) => p },
    );
    initFn = Module.cwrap('engine_init', 'number', []);
    initBytesFn = Module.cwrap('engine_init_with_model_bytes', 'number', ['number', 'number', 'number']);
    reloadBytesFn = Module.cwrap('engine_reload_model_bytes', 'number', ['number', 'number', 'number']);
    inferFn = Module.cwrap('engine_infer', 'number', ['number', 'string', 'number', 'number', 'number']);
    modelInfoFn = Module.cwrap('engine_model_info', 'string', ['number']);
    cleanupFn = Module.cwrap('engine_cleanup', null, ['number']);
    if (!handle) handle = initFn();
    return true;
  }

  function _disposeHandle() {
    if (handle && cleanupFn) {
      cleanupFn(handle);
      handle = 0;
    }
  }

  function loadModel(arrayBuffer) {
    if (!Module || !initBytesFn) throw new Error('Call init() first');
    if (arrayBuffer.byteLength > MAX_MODEL_BYTES) {
      throw new Error(`Model exceeds ${MAX_MODEL_BYTES / (1024 * 1024)} MB browser cap`);
    }
    const bytes = new Uint8Array(arrayBuffer);
    const ptr = Module._malloc(bytes.length);
    Module.HEAPU8.set(bytes, ptr);
    _disposeHandle();
    handle = initBytesFn(ptr, bytes.length, 0);
    Module._free(ptr);
    if (!handle) throw new Error('Failed to load .nanoq from buffer');
    return modelInfo();
  }

  function modelInfo() {
    if (!handle || !modelInfoFn) return {};
    try {
      return JSON.parse(modelInfoFn(handle) || '{}');
    } catch (_) {
      return {};
    }
  }

  function infer(prompt, opts = {}) {
    if (!handle || !inferFn) throw new Error('Engine not ready');
    const maxTokens = opts.maxTokens || 24;
    const outPtr = Module._malloc(OUT_BUF);
    const t0 = performance.now();
    inferFn(handle, prompt, maxTokens, outPtr, OUT_BUF);
    const text = Module.UTF8ToString(outPtr);
    Module._free(outPtr);
    return { text, latencyMs: performance.now() - t0, warnings: [] };
  }

  function dispose() {
    _disposeHandle();
    Module = null;
  }

  return { init, loadModel, modelInfo, infer, dispose, MAX_MODEL_BYTES };
})();

if (typeof module !== 'undefined') module.exports = NanoServeWasm;
