/* CardioCare workstation front end.
   The rhythm strip is drawn on a real calibrated grid: 25 mm/s and 10 mm/mV by
   default, small square 1 mm, heavy line every 5 mm. R peaks are marked with
   caliper ticks and the RR interval is written between them, the way a
   clinician would step through a strip with dividers. */

const $ = (s) => document.querySelector(s);

const PAPER = {
  bg: '#FDF6F3', fine: '#F2C7BD', bold: '#E0897A',
  ink: '#14181C', mark: '#0B7A6B', muted: '#8496A3',
};

const SEV_COLOR = {
  normal: '#0B7A6B', monitor: '#C79213', urgent: '#B3341F', critical: '#7A160B',
};

let LAST = null;    // most recent analysis payload
let BUSY = false;

/* ------------------------------------------------------------- helpers */

function num(v, nd = 0, unit = '') {
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  return Number(v).toFixed(nd) + unit;
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function setBusy(on, msg) {
  BUSY = on;
  $('#input-msg').innerHTML = on
    ? `<span class="spin"></span> <span style="font-size:12.5px;color:var(--text-2)">${esc(msg || 'Analysing…')}</span>`
    : '';
  document.querySelectorAll('.chip, .btn').forEach((b) => { b.disabled = on; });
}

function showError(msg) {
  $('#input-msg').innerHTML = `<div class="err">${esc(msg)}</div>`;
}

/* ------------------------------------------------------ strip rendering */

function drawStrip(payload) {
  const cv = $('#strip');
  const wave = payload.waveform;
  const vals = wave.values;
  if (!vals || vals.length < 2) return;

  const dpr = window.devicePixelRatio || 1;
  const speed = Number($('#speed').value) || 25;   // mm/s
  const gain = Number($('#gain').value) || 10;     // mm/mV

  // 4.4 CSS px per mm gives a legible strip without needing to scroll far.
  const pxmm = 4.4;
  const heightMm = 300 / pxmm;
  const durationS = wave.duration_s;
  const widthMm = durationS * speed + 8;
  const cssW = Math.max(760, widthMm * pxmm);
  const cssH = 300;

  cv.style.width = cssW + 'px';
  cv.style.height = cssH + 'px';
  cv.width = Math.round(cssW * dpr);
  cv.height = Math.round(cssH * dpr);

  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);

  g.fillStyle = PAPER.bg;
  g.fillRect(0, 0, cssW, cssH);

  // Grid
  g.lineWidth = 1;
  g.strokeStyle = PAPER.fine;
  g.beginPath();
  for (let x = 0; x <= cssW; x += pxmm) { g.moveTo(Math.round(x) + 0.5, 0); g.lineTo(Math.round(x) + 0.5, cssH); }
  for (let y = 0; y <= cssH; y += pxmm) { g.moveTo(0, Math.round(y) + 0.5); g.lineTo(cssW, Math.round(y) + 0.5); }
  g.stroke();

  g.strokeStyle = PAPER.bold;
  g.beginPath();
  for (let x = 0; x <= cssW; x += pxmm * 5) { g.moveTo(Math.round(x) + 0.5, 0); g.lineTo(Math.round(x) + 0.5, cssH); }
  for (let y = 0; y <= cssH; y += pxmm * 5) { g.moveTo(0, Math.round(y) + 0.5); g.lineTo(cssW, Math.round(y) + 0.5); }
  g.stroke();

  const baseY = cssH / 2;
  const pxPerSec = speed * pxmm;
  const pxPerMv = gain * pxmm;
  const left = 4 * pxmm;

  // Calibration pulse: 10 mm tall, 5 mm wide, exactly as printed on real paper.
  g.strokeStyle = PAPER.ink;
  g.lineWidth = 1.6;
  g.lineJoin = 'round';
  g.beginPath();
  g.moveTo(2, baseY);
  g.lineTo(left - 5 * pxmm, baseY);
  g.lineTo(left - 5 * pxmm, baseY - 10 * pxmm);
  g.lineTo(left, baseY - 10 * pxmm);
  g.lineTo(left, baseY);
  g.stroke();

  // Trace
  const xAt = (i) => left + 2 * pxmm + (i / wave.fs) * pxPerSec;
  const yAt = (v) => Math.max(2, Math.min(cssH - 2, baseY - v * pxPerMv));

  g.beginPath();
  g.moveTo(xAt(0), yAt(vals[0]));
  for (let i = 1; i < vals.length; i++) g.lineTo(xAt(i), yAt(vals[i]));
  g.strokeStyle = PAPER.ink;
  g.lineWidth = 1.6;
  g.stroke();

  // R-peak calipers and RR intervals
  const peaks = wave.r_peaks || [];
  if (peaks.length) {
    g.strokeStyle = PAPER.mark;
    g.fillStyle = PAPER.mark;
    g.lineWidth = 1;
    g.font = '500 10px "IBM Plex Mono", monospace';
    g.textAlign = 'center';

    peaks.forEach((p) => {
      if (p < 0 || p >= vals.length) return;
      const x = xAt(p);
      g.beginPath();
      g.moveTo(x, 6); g.lineTo(x, 15);
      g.stroke();
    });

    for (let i = 1; i < peaks.length; i++) {
      const a = peaks[i - 1], b = peaks[i];
      if (b >= vals.length) break;
      const x1 = xAt(a), x2 = xAt(b);
      if (x2 - x1 < 34) continue;
      const rr = Math.round(((b - a) / wave.fs) * 1000);
      g.beginPath();
      g.moveTo(x1, 21); g.lineTo(x2, 21);
      g.stroke();
      g.fillStyle = PAPER.bg;
      g.fillRect((x1 + x2) / 2 - 17, 15, 34, 12);
      g.fillStyle = PAPER.mark;
      g.fillText(rr + 'ms', (x1 + x2) / 2, 25);
    }
  }

  $('#strip-wrap').hidden = false;
  $('#strip-empty').hidden = true;

  const d = payload.source?.digitization;
  $('#cal').hidden = false;
  $('#cal').innerHTML = [
    `<span><b>${speed}</b> mm/s</span>`,
    `<span><b>${gain}</b> mm/mV</span>`,
    `<span><b>${durationS.toFixed(1)}</b> s</span>`,
    `<span><b>${peaks.length}</b> beats marked</span>`,
    d ? `<span>digitised at <b>${d.dpi}</b> DPI · <b>${Math.round(d.coverage * 100)}%</b> trace coverage</span>` : '',
    `<span>analysed in <b>${payload.timing_ms}</b> ms</span>`,
  ].filter(Boolean).join('');
}

/* --------------------------------------------------------- result panes */

function renderVerdict(f) {
  const sev = f.inconclusive ? 'unknown' : f.severity;
  const col = f.inconclusive ? '#8496A3' : (SEV_COLOR[f.severity] || '#14181C');
  const pct = Math.round((f.confidence || 0) * 100);

  const badge = {
    corroborated: '<span class="badge ok">✓ corroborated by measurements</span>',
    discordant: '<span class="badge bad">⚠ model and rules disagree</span>',
    rules_only: '<span class="badge info">measurement rules only</span>',
  }[f.agreement] || '';

  const title = f.inconclusive
    ? 'Inconclusive'
    : esc(f.name);

  const sub = f.inconclusive
    ? `Highest-scoring class is ${esc(f.name)} at ${pct}%, below the reporting threshold. Manual review required.`
    : esc(f.agreement_note || '');

  $('#verdict').innerHTML = `
    <div class="verdict ${sev}">
      <div class="lbl">Automated finding</div>
      <h2 style="color:${col}">${title}</h2>
      <div>${badge}</div>
      <p class="desc">${sub}</p>
      <div class="meter">
        <div class="row"><span>CONFIDENCE</span><span>${pct}%</span></div>
        <div class="bar"><i style="width:${pct}%;background:${col}"></i></div>
      </div>
    </div>`;
}

function renderVitals(m) {
  const wideAlert = (m.qrs_duration_ms || 0) > 120;
  const hr = m.heart_rate_bpm;
  const hrAlert = hr !== null && (hr < 60 || hr > 100);

  const cells = [
    ['Heart rate', num(hr, 0), 'bpm', hrAlert],
    ['QRS', num(m.qrs_duration_ms, 0), 'ms', wideAlert],
    ['RR mean', num(m.rr_mean_ms, 0), 'ms', false],
    ['RR variation', num(m.rr_cv, 3), '', (m.rr_cv || 0) > 0.13],
    ['RMSSD', num(m.rmssd_ms, 0), 'ms', false],
    ['Beats', num(m.beat_count, 0), '', false],
    ['P waves', m.p_wave_present ? 'yes' : 'no', '', !m.p_wave_present],
    ['Atrial rate', m.atrial_activity_organised ? num(m.atrial_rate_bpm, 0) : null, '/min', false],
    ['Wide beats', num((m.qrs_wide_fraction ?? 0) * 100, 0), '%', (m.qrs_wide_fraction || 0) > 0.15],
    ['Signal quality', num((m.signal_quality ?? 0) * 100, 0), '%', (m.signal_quality ?? 1) < 0.6],
  ];

  $('#vitals').innerHTML = cells.map(([k, v, u, alert]) => `
    <div class="vital">
      <div class="k">${k}</div>
      <div class="v ${v === null ? 'na' : (alert ? 'alert' : '')}">${
        v === null ? 'n/a' : esc(v) + (u ? `<span class="u">${u}</span>` : '')
      }</div>
    </div>`).join('');
}

function renderEvidence(f, m) {
  const yes = (f.evidence || []).map((e) => `<li>${esc(e)}</li>`).join('');
  const no = (f.contradicting || []).map((e) => `<li>${esc(e)}</li>`).join('');
  let html = yes ? `<ul class="ev">${yes}</ul>` : '<p style="font-size:12.5px;color:var(--text-3)">No criteria met.</p>';
  if (no) {
    html += `<div style="font-family:var(--mono);font-size:9.5px;letter-spacing:.1em;
             text-transform:uppercase;color:var(--text-3);margin:13px 0 3px">Not met</div>
             <ul class="ev no">${no}</ul>`;
  }
  const warns = m.quality_notes || [];
  if (warns.length) {
    html += `<div class="warnbox"><b>Signal quality</b><ul>${
      warns.map((w) => `<li>${esc(w)}</li>`).join('')}</ul></div>`;
  }
  $('#evidence').innerHTML = html;
}

function renderDifferential(f) {
  $('#differential').innerHTML = `<ul class="dx">${
    (f.differential || []).map((d) => {
      const pct = (d.probability * 100).toFixed(1);
      const col = SEV_COLOR[d.severity] || '#8496A3';
      return `<li>
        <div class="row"><span>${esc(d.name)}</span><span class="pct">${pct}%</span></div>
        <div class="bar"><i style="width:${Math.max(1.5, d.probability * 100)}%;background:${col}"></i></div>
      </li>`;
    }).join('')}</ul>`;
}

function renderProvenance(p) {
  const f = p.finding, s = p.source || {}, d = s.digitization;
  const rows = [
    ['Analysis path', (f.agreement || '').replace(/_/g, ' ')],
    ['Model', f.model?.available ? `${esc(f.model.name)} · sha256 ${esc(f.model.sha256)}` : 'none loaded'],
    f.model?.available ? ['Trained on', {
      physionet: 'PhysioNet recordings',
      synthetic: 'synthetic waveforms — not clinically meaningful',
      unknown: 'unknown provenance',
    }[f.model.training_data] || esc(f.model.training_data)] : null,
    ['Model weight', f.model?.available ? `${Math.round((f.model.cnn_weight || 0) * 100)}% CNN / ${Math.round((1 - (f.model.cnn_weight || 0)) * 100)}% rules` : '0% — rules only'],
    ['Input', s.type === 'sample' ? `reference strip “${esc(s.name)}”` : `${esc(s.type || 'unknown')} · ${esc(s.filename || '')}`],
    s.ground_truth ? ['Ground truth', esc(s.ground_truth)] : null,
    d ? ['Digitisation', `${d.dpi} DPI · ${d.px_per_mm} px/mm · rotation ${d.rotation_deg}° · coverage ${Math.round(d.coverage * 100)}%`] : null,
    ['Analysed', new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC'],
  ].filter(Boolean);

  let html = `<table style="width:100%;border-collapse:collapse;font-size:12.5px">${
    rows.map(([k, v]) => `<tr>
      <td style="padding:5px 12px 5px 0;color:var(--text-3);white-space:nowrap;
                 font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;
                 text-transform:uppercase;vertical-align:top">${k}</td>
      <td style="padding:5px 0">${v}</td></tr>`).join('')}</table>`;

  const warns = d?.warnings || [];
  if (warns.length) {
    html += `<div class="warnbox"><b>Digitisation warnings</b><ul>${
      warns.map((w) => `<li>${esc(w)}</li>`).join('')}</ul></div>`;
  }
  if (f.model?.available && f.model.training_data !== 'physionet') {
    html += `<div class="warnbox" style="margin-top:10px">
      <b>${f.model.training_data === 'synthetic' ? 'Synthetic weights.' : 'Unverified weights.'}</b>
      ${esc(f.model.note || '')}</div>`;
  }
  if (f.model && !f.model.available) {
    html += `<div class="warnbox" style="margin-top:10px">
      <b>No trained model loaded.</b> ${esc(f.model.note || '')}
      This finding comes from the measurement rule engine alone. Train a model with
      <code>ml/train.py</code> and place the exported <code>.onnx</code> file in
      <code>models/</code> to enable the second path.</div>`;
  }
  $('#provenance').innerHTML = html;
}

function render(payload) {
  LAST = payload;
  drawStrip(payload);
  renderVerdict(payload.finding);
  renderVitals(payload.measurements);
  renderEvidence(payload.finding, payload.measurements);
  renderDifferential(payload.finding);
  renderProvenance(payload);
  $('#results').hidden = false;
  $('#strip-src').textContent =
    payload.source?.type === 'sample' ? 'REFERENCE' : 'UPLOADED';
}

/* -------------------------------------------------------------- network */

async function post(url, body, label) {
  if (BUSY) return;
  setBusy(true, label);
  try {
    const res = await fetch(url, { method: 'POST', body });
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch { data = null; }
    if (!res.ok) throw new Error(data?.detail || `Request failed (${res.status}).`);
    setBusy(false);
    render(data);
  } catch (e) {
    setBusy(false);
    showError(e.message || 'Analysis failed.');
  }
}

function analyseSample(name, btn) {
  document.querySelectorAll('.chip').forEach((c) => c.setAttribute('aria-pressed', 'false'));
  if (btn) btn.setAttribute('aria-pressed', 'true');
  const fd = new FormData();
  fd.append('name', name);
  post('/api/analyze/sample', fd, 'Analysing reference strip…');
}

function analyseFile(file) {
  if (!file) return;
  const isImage = file.type.startsWith('image/') ||
    /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(file.name);
  const fd = new FormData();
  fd.append('file', file);
  if (isImage) {
    fd.append('paper_speed', $('#speed').value);
    fd.append('gain', $('#gain').value);
    post('/api/analyze/image', fd, `Digitising ${file.name}…`);
  } else {
    fd.append('fs', '0');
    post('/api/analyze/signal', fd, `Reading ${file.name}…`);
  }
}

async function exportPdf() {
  if (!LAST) return;
  const fd = new FormData();
  fd.append('payload', JSON.stringify(LAST));
  const res = await fetch('/api/report', { method: 'POST', body: fd });
  if (!res.ok) { showError('Could not generate the report.'); return; }
  const blob = await res.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'cardiocare-report.pdf';
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ------------------------------------------------------------- startup */

async function init() {
  try {
    const h = await (await fetch('/api/health')).json();
    const ok = h.model?.available;
    $('#dot-model').className = 'dot ' + (ok ? 'on' : 'off');
    $('#stat-model').textContent = ok
      ? `model ${h.model.name} · ${h.model.sha256}`
      : 'no model — rules only';
  } catch {
    $('#dot-model').className = 'dot off';
    $('#stat-model').textContent = 'service unreachable';
  }

  try {
    const s = await (await fetch('/api/samples')).json();
    const list = s.samples || [];
    $('#chips').innerHTML = list.length
      ? list.map((x) => `
          <button class="chip" data-name="${esc(x.key)}" aria-pressed="false"
                  title="${esc(x.signature || '')}">
            <span class="sw ${esc(x.severity)}"></span>
            <span class="nm">${esc(x.short)}</span>
            <span class="cd">${esc(x.key.slice(0, 6))}</span>
          </button>`).join('')
      : '<div style="font-size:12.5px;color:var(--text-3)">Run <code>python ml/make_samples.py</code> to build the gallery.</div>';

    document.querySelectorAll('.chip').forEach((b) =>
      b.addEventListener('click', () => analyseSample(b.dataset.name, b)));
  } catch {
    $('#chips').innerHTML = '<div class="err">Could not load reference strips.</div>';
  }

  const drop = $('#drop'), input = $('#file');
  drop.addEventListener('click', () => input.click());
  drop.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  ['dragenter', 'dragover'].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('hot'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('hot'); }));
  drop.addEventListener('drop', (e) => analyseFile(e.dataTransfer.files[0]));
  input.addEventListener('change', () => analyseFile(input.files[0]));

  $('#btn-image').addEventListener('click', () => { input.accept = 'image/*'; input.click(); });
  $('#btn-signal').addEventListener('click', () => { input.accept = '.csv,.tsv,.txt,.json'; input.click(); });
  $('#btn-pdf').addEventListener('click', exportPdf);

  ['#speed', '#gain'].forEach((s) =>
    $(s).addEventListener('change', () => { if (LAST) drawStrip(LAST); }));
}

init();
