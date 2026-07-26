async function refreshHealth() {
  const gpu = document.getElementById('chipGpu');
  const models = document.getElementById('chipModels');
  const gguf = document.getElementById('chipGguf');
  try {
    const res = await fetch('/health');
    const data = await res.json();
    document.getElementById('chipStatus').querySelector('.chip-dot').className =
      'chip-dot' + (data.status === 'ok' ? '' : ' warn');

    const gpuOn = data.gpu_available;
    gpu.querySelector('.chip-dot').className = 'chip-dot' + (gpuOn ? '' : ' off');
    gpu.lastChild.textContent = gpuOn ? 'GPU ready' : 'CPU only';

    models.lastChild.textContent = `${data.models_registered || 0} models`;

    const ggufOn = data.gguf_available;
    gguf.querySelector('.chip-dot').className = 'chip-dot' + (ggufOn ? '' : ' off');
    gguf.lastChild.textContent = ggufOn ? 'GGUF on' : 'GGUF off';
  } catch (_) {
    document.getElementById('chipStatus').querySelector('.chip-dot').className = 'chip-dot warn';
  }
}

async function refreshModels() {
  const sel = document.getElementById('model');
  const current = sel.value;
  sel.innerHTML = '<option value="">Default</option>';
  try {
    const res = await fetch('/v1/models');
    const data = await res.json();
    for (const m of data.models || []) {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = `${m.id} · ${m.dtype}${m.quantized ? '' : ' (raw)'}`;
      sel.appendChild(opt);
    }
    if ([...sel.options].some(o => o.value === current)) sel.value = current;
  } catch (_) {}
}

function setMetaTags(data) {
  const bar = document.getElementById('meta');
  bar.innerHTML = '';
  const tags = [
    ['device', data.device],
    ['format', data.format],
    ['quantized', data.quantized ? 'yes' : 'no'],
    ['latency', `${data.latency_ms.toFixed(1)} ms`],
  ];
  if (data.model) tags.unshift(['model', data.model]);
  if (data.id) tags.push(['id', data.id.slice(0, 8)]);

  for (const [k, v] of tags) {
    const el = document.createElement('span');
    el.className = 'meta-tag';
    el.innerHTML = `<strong>${k}</strong> ${v}`;
    bar.appendChild(el);
  }
  if (data.warnings?.length) {
    for (const w of data.warnings) {
      const el = document.createElement('span');
      el.className = 'meta-tag';
      el.style.borderColor = 'rgba(255, 214, 10, 0.35)';
      el.innerHTML = `<strong>warn</strong> ${w}`;
      bar.appendChild(el);
    }
  }
}

document.getElementById('downloadBtn').onclick = () =>
  document.getElementById('dlDialog').showModal();
document.getElementById('dlCancel').onclick = () =>
  document.getElementById('dlDialog').close();

document.getElementById('dlConfirm').onclick = async () => {
  const source = document.getElementById('dlSource').value;
  const target = document.getElementById('dlTarget').value.trim();
  if (!target) return alert('Enter a repo ID or URL');
  const body = { source, precision: document.getElementById('precision').value };
  if (source === 'hf') body.repo_id = target;
  else body.url = target;
  const fn = document.getElementById('dlFilename').value.trim();
  const mid = document.getElementById('dlModelId').value.trim();
  if (fn) body.filename = fn;
  if (mid) body.model_id = mid;

  const btn = document.getElementById('dlConfirm');
  btn.disabled = true;
  btn.textContent = 'Downloading…';
  try {
    const res = await fetch('/v1/models/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    document.getElementById('dlDialog').close();
    await refreshModels();
    await refreshHealth();
    if (data.model) document.getElementById('model').value = data.model.id;
    setMetaTags({
      device: '—', format: data.format, model: data.model?.id,
      quantized: data.quantized, latency_ms: 0, id: 'download',
      warnings: data.warnings,
    });
  } catch (e) {
    alert('Download failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Download';
  }
};

const go = document.getElementById('go');
go.onclick = async () => {
  const out = document.getElementById('out');
  const prompt = document.getElementById('prompt').value;
  const max_tokens = parseInt(document.getElementById('tokens').value || '24', 10);
  const device = document.getElementById('device').value;
  const model = document.getElementById('model').value || null;
  const format = document.getElementById('format').value;
  const precision = document.getElementById('precision').value;

  go.disabled = true;
  go.textContent = 'Generating…';
  out.textContent = 'Thinking…';
  out.className = 'output-text loading';
  document.getElementById('meta').innerHTML = '';

  try {
    const payload = { prompt, max_tokens, device, format, precision };
    if (model) payload.model = model;
    if (precision === 'raw') payload.quantize = false;

    const res = await fetch('/v1/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);

    out.textContent = data.text;
    out.className = 'output-text';
    setMetaTags(data);
  } catch (e) {
    out.textContent = String(e.message || e);
    out.className = 'output-text error';
  } finally {
    go.disabled = false;
    go.textContent = 'Generate';
  }
};

refreshHealth();
refreshModels();
