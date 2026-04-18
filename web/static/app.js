'use strict';

const DISEASES = ['Diabetes', 'MS', 'Glaucoma', 'DryEye', 'Healthy'];
const DISEASE_LABELS = {
  Diabetes: 'Diabetes', MS: 'MS', Glaucoma: 'Glaucoma',
  DryEye: 'Dry Eye', Healthy: 'Healthy',
};
const CIRC = 87.96; // 2π × r=14
const MODEL_POLL_INTERVAL_MS = 350;

let results = [];
let activeFilter = 'all';
let classifying = false;
let queuedFiles = [];
let modelReady = false;

// ── Risk helpers ────────────────────────────────────────────
function pctLevel(pct) {
  if (pct >= 0.6) return 'high';
  if (pct >= 0.2) return 'warning';
  return 'safe';
}

function cardLevel(probs) {
  const nonHealthy = DISEASES.filter(d => d !== 'Healthy').map(d => probs[d]);
  if (nonHealthy.some(p => p >= 0.6)) return 'high';
  if (nonHealthy.some(p => p >= 0.2)) return 'warning';
  return 'safe';
}

const LEVEL_ORDER = { high: 0, warning: 1, safe: 2 };
const LEVEL_COLOR = { high: '#b82c2c', warning: '#c07a18', safe: '#3a8858' };
const BADGE_CLASS = { high: 'risk-high', warning: 'risk-warn', safe: 'risk-safe' };
const BADGE_LABEL = { high: 'High Risk', warning: 'Warning', safe: 'Healthy' };

// ── SVG arc gauge ───────────────────────────────────────────
function arcSVG(pct) {
  const offset = (CIRC * (1 - pct)).toFixed(2);
  const color = LEVEL_COLOR[pctLevel(pct)];
  const label = Math.round(pct * 100) + '%';
  return `<svg viewBox="0 0 36 36">
    <circle cx="18" cy="18" r="14" fill="none" stroke="#ece6de" stroke-width="3.5"/>
    <circle cx="18" cy="18" r="14" fill="none" stroke="${color}" stroke-width="3.5"
      stroke-dasharray="${CIRC}" stroke-dashoffset="${offset}"
      stroke-linecap="round" transform="rotate(-90 18 18)"/>
    <text x="18" y="21" text-anchor="middle" font-size="9.5" fill="#1a1410"
      font-weight="700" font-family="Segoe UI,sans-serif">${label}</text>
  </svg>`;
}

// ── Card HTML ───────────────────────────────────────────────
function makeCard(r) {
  const level = r.riskLevel;
  const topDisease = DISEASES.reduce((best, disease) =>
    r.probabilities[disease] > r.probabilities[best] ? disease : best
  , DISEASES[0]);
  const topPct = Math.round((r.probabilities[topDisease] || 0) * 100);
  const arcs = DISEASES.map(d => `
    <div class="arc-item">
      ${arcSVG(r.probabilities[d])}
      <span class="arc-label">${DISEASE_LABELS[d]}</span>
    </div>`).join('');
  return `<div class="result-card" data-risk="${level}">
    <div class="card-image-wrap"
         data-src="${r.objectUrl}"
         data-caption="${r.filename}">
      <img src="${r.objectUrl}" alt="${r.filename}">
      <div class="risk-overlay ${BADGE_CLASS[level]}">${BADGE_LABEL[level]}</div>
    </div>
    <div class="card-meta">
      <span class="card-filename">${r.filename}</span>
    </div>
    <div class="card-body">
      <div class="top-prediction">
        <span class="top-prediction-label">Top prediction</span>
        <strong class="top-prediction-value">${DISEASE_LABELS[topDisease]} (${topPct}%)</strong>
      </div>
      <div class="arcs-grid">${arcs}</div>
    </div>
  </div>`;
}

// ── Render ──────────────────────────────────────────────────
function render() {
  const sorted = [...results].sort(
    (a, b) => LEVEL_ORDER[a.riskLevel] - LEVEL_ORDER[b.riskLevel]
  );
  const visible = activeFilter === 'all'
    ? sorted
    : sorted.filter(r => r.riskLevel === activeFilter);
  document.getElementById('results-grid').innerHTML = visible.map(makeCard).join('');
  document.getElementById('count-total').textContent = results.length;
  document.getElementById('count-high').textContent =
    results.filter(r => r.riskLevel === 'high').length;
  document.getElementById('count-warning').textContent =
    results.filter(r => r.riskLevel === 'warning').length;
  document.getElementById('count-safe').textContent =
    results.filter(r => r.riskLevel === 'safe').length;
}

function setRunnerProgress(done, total) {
  const wrap = document.getElementById('classification-progress');
  const count = document.getElementById('runner-count');
  const bar = document.getElementById('runner-bar');
  const pct = total ? Math.round((done / total) * 100) : 0;
  wrap.hidden = false;
  count.textContent = `${done} / ${total}`;
  bar.style.width = `${pct}%`;
}

function hideRunnerProgress() {
  const wrap = document.getElementById('classification-progress');
  const count = document.getElementById('runner-count');
  const bar = document.getElementById('runner-bar');
  count.textContent = '0 / 0';
  bar.style.width = '0%';
  wrap.hidden = true;
}

function renderModelLoader(status) {
  const overlay = document.getElementById('model-loader');
  const stage = document.getElementById('model-loader-stage');
  const value = document.getElementById('model-loader-value');
  const bar = document.getElementById('model-loader-bar');

  const progress = Math.max(0, Math.min(100, Number(status.progress || 0)));
  stage.textContent = status.stage || 'Loading model';
  value.textContent = `${progress}%`;
  bar.style.width = `${progress}%`;

  if (status.loaded) {
    modelReady = true;
    overlay.hidden = true;
  } else {
    overlay.hidden = false;
  }
}

async function ensureModelReady() {
  if (modelReady) return;

  const overlay = document.getElementById('model-loader');
  overlay.hidden = false;

  while (!modelReady) {
    try {
      const resp = await fetch('/api/model-status');
      if (resp.ok) {
        const status = await resp.json();
        renderModelLoader(status);
      } else {
        renderModelLoader({ stage: 'Preparing model', progress: 5, loaded: false });
      }
    } catch (_) {
      renderModelLoader({ stage: 'Waiting for server', progress: 5, loaded: false });
    }

    if (modelReady) break;
    await new Promise(resolve => setTimeout(resolve, MODEL_POLL_INTERVAL_MS));
  }
}

// ── Classification ──────────────────────────────────────────
async function classifyFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  const resp = await fetch('/classify', { method: 'POST', body: fd });
  if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`);
  return resp.json();
}

async function runClassification(files) {
  if (classifying || !files.length) return;

  await ensureModelReady();

  classifying = true;
  results = [];

  const resultsSection = document.getElementById('results-section');

  resultsSection.hidden = false;
  setRunnerProgress(0, files.length);

  let finished = 0;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];

    try {
      const data = await classifyFile(file);
      results.push({
        filename: file.name,
        probabilities: data.probabilities,
        riskLevel: cardLevel(data.probabilities),
        objectUrl: URL.createObjectURL(file),
      });
      render();
    } catch (err) {
      console.error('classify failed:', file.name, err);
    } finally {
      finished += 1;
      setRunnerProgress(finished, files.length);
    }
  }
  hideRunnerProgress();
  classifying = false;
}

// ── File queue ──────────────────────────────────────────────
function updateChips() {
  document.getElementById('file-chips').innerHTML =
    queuedFiles.map(f => `<div class="queued-file">${f.name}</div>`).join('');
}

function isBmp(file) {
  return file.name.toLowerCase().endsWith('.bmp');
}

function addFiles(list) {
  const arr = Array.from(list);
  const bmps = arr.filter(isBmp);
  const rejected = arr.length - bmps.length;
  if (rejected > 0) {
    alert(`${rejected} file(s) skipped — only .bmp images are supported.`);
  }
  queuedFiles = [...queuedFiles, ...bmps];
  updateChips();
}

// ── Filter ──────────────────────────────────────────────────
function setFilter(level) {
  activeFilter = level;
  document.querySelectorAll('.filter-btn').forEach(btn =>
    btn.classList.toggle('active', btn.dataset.filter === level));
  render();
}

// ── Lightbox ────────────────────────────────────────────────
function openLightbox(src, caption) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox-caption').textContent = caption;
  document.getElementById('lightbox').hidden = false;
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  document.getElementById('lightbox').hidden = true;
  document.body.style.overflow = '';
}

// ── Demo mode ───────────────────────────────────────────────
async function loadDemoScans() {
  const resp = await fetch('/api/demo-scans');
  const { filenames } = await resp.json();
  if (!filenames.length) {
    alert('No demo scans found. Add BMP files to web/demo_scans/.');
    return;
  }
  const files = await Promise.all(filenames.map(async name => {
    const r = await fetch(`/demo-scans/${name}`);
    const blob = await r.blob();
    return new File([blob], name, { type: 'image/bmp' });
  }));
  queuedFiles = files;
  updateChips();
  await runClassification(files);
}

// ── Init ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const uploadZone = document.getElementById('upload-zone');
  const fileInput  = document.getElementById('file-input');

  uploadZone.addEventListener('click', e => {
    const target = e.target;
    if (!(target instanceof Element)) return;

    if (target.closest('#analyse-btn, button, a, input, select, textarea, [role="button"]')) {
      return;
    }

    fileInput.click();
  });
  fileInput.addEventListener('change', e => addFiles(e.target.files));

  uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });
  uploadZone.addEventListener('dragleave', () =>
    uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    addFiles(e.dataTransfer.files);
  });

  document.getElementById('analyse-btn')
    .addEventListener('click', e => {
      e.preventDefault();
      e.stopPropagation();
      runClassification(queuedFiles);
    });
  document.getElementById('demo-btn')
    .addEventListener('click', loadDemoScans);
  document.getElementById('export-btn')
    .addEventListener('click', () => window.print());

  document.querySelectorAll('.filter-btn').forEach(btn =>
    btn.addEventListener('click', () => setFilter(btn.dataset.filter)));

  // Lightbox — event delegation so dynamically-added cards work
  document.getElementById('results-grid').addEventListener('click', e => {
    const wrap = e.target.closest('.card-image-wrap');
    if (wrap) openLightbox(wrap.dataset.src, wrap.dataset.caption);
  });

  const lightbox = document.getElementById('lightbox');
  document.getElementById('lightbox-close').addEventListener('click', closeLightbox);
  lightbox.addEventListener('click', e => { if (e.target === lightbox) closeLightbox(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

  ensureModelReady();
});
