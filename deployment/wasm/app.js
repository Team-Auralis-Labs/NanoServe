async function refreshStatus(msg, ok = true) {
  const chip = document.getElementById('chipStatus');
  chip.querySelector('.chip-dot').className = 'chip-dot' + (ok ? '' : ' warn');
  chip.lastChild.textContent = msg;
}

function setMeta(data) {
  const bar = document.getElementById('meta');
  bar.innerHTML = '';
  const tags = [
    ['format', 'nanoq'],
    ['latency', `${data.latencyMs.toFixed(1)} ms`],
  ];
  if (data.modelName) tags.unshift(['model', data.modelName]);
  if (data.dtype) tags.unshift(['dtype', data.dtype]);
  for (const [k, v] of tags) {
    const el = document.createElement('span');
    el.className = 'meta-tag';
    el.innerHTML = `<strong>${k}</strong> ${v}`;
    bar.appendChild(el);
  }
}

let modelLabel = '';

document.getElementById('loadBtn').onclick = () =>
  document.getElementById('fileInput').click();

document.getElementById('fileInput').onchange = async (ev) => {
  const file = ev.target.files?.[0];
  if (!file) return;
  if (!file.name.endsWith('.nanoq')) {
    alert('Please select a .nanoq file');
    return;
  }
  try {
    const buf = await file.arrayBuffer();
    const info = NanoServeWasm.loadModel(buf);
    modelLabel = file.name;
    document.getElementById('chipModel').lastChild.textContent = file.name;
    document.getElementById('chipModel').querySelector('.chip-dot').className = 'chip-dot';
    refreshStatus('Model loaded');
    setMeta({ latencyMs: 0, modelName: file.name, dtype: info.dtype || '?' });
  } catch (e) {
    alert(String(e.message || e));
    refreshStatus('Load failed', false);
  }
};

const go = document.getElementById('go');
go.onclick = async () => {
  const out = document.getElementById('out');
  const prompt = document.getElementById('prompt').value;
  const maxTokens = parseInt(document.getElementById('tokens').value || '24', 10);

  if (!modelLabel) {
    out.textContent = 'Load a .nanoq file first (Load .nanoq file button).';
    out.className = 'output-text error';
    return;
  }

  go.disabled = true;
  go.textContent = 'Generating…';
  out.textContent = 'Thinking…';
  out.className = 'output-text loading';

  try {
    const data = NanoServeWasm.infer(prompt, { maxTokens });
    out.textContent = data.text;
    out.className = 'output-text';
    const info = NanoServeWasm.modelInfo();
    setMeta({ ...data, modelName: modelLabel, dtype: info.dtype });
  } catch (e) {
    out.textContent = String(e.message || e);
    out.className = 'output-text error';
  } finally {
    go.disabled = false;
    go.textContent = 'Generate';
  }
};

(async () => {
  try {
    await NanoServeWasm.init();
    refreshStatus('WASM ready');
    const demo = './assets/demo.nanoq';
    try {
      const res = await fetch(demo);
      if (res.ok) {
        const info = NanoServeWasm.loadModel(await res.arrayBuffer());
        modelLabel = 'demo.nanoq';
        document.getElementById('chipModel').lastChild.textContent = 'demo.nanoq';
        document.getElementById('chipModel').querySelector('.chip-dot').className = 'chip-dot';
        setMeta({ latencyMs: 0, modelName: 'demo.nanoq', dtype: info.dtype });
      }
    } catch (_) {
      refreshStatus('WASM ready — load a .nanoq file');
    }
  } catch (e) {
    refreshStatus('WASM load failed', false);
    document.getElementById('out').textContent =
      'Build WASM first: ./scripts/build_wasm.sh\n\n' + String(e.message || e);
    document.getElementById('out').className = 'output-text error';
  }
})();
