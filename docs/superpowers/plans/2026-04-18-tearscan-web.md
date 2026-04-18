# TearScan Web App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI + vanilla JS web app for classifying dried tear drop AFM scans with multi-upload, arc-gauge results, risk sorting/filtering, lightbox, demo mode, and PDF export.

**Architecture:** FastAPI serves both the `/classify` API endpoint and the static frontend files. The frontend is a single HTML/CSS/JS SPA with no build tooling. Images are classified sequentially; result cards appear as they complete, always sorted high → warning → clear.

**Tech Stack:** Python 3.14, FastAPI, uvicorn, python-multipart, Pillow; Vanilla HTML/CSS/JS (no framework).

---

## File Map

| File | Purpose |
|------|---------|
| `web/main.py` | FastAPI app: `/classify`, `/api/demo-scans`, static mounts |
| `web/classifier.py` | Stub wrapper — replace with real model |
| `web/static/index.html` | Full HTML shell |
| `web/static/styles.css` | All styles incl. print/PDF rules |
| `web/static/app.js` | All JavaScript |
| `web/demo_scans/` | 5 sample BMPs (one per class) |

---

## Task 1: Project Setup

**Files:**
- Modify: `pyproject.toml`
- Create: `web/`, `web/static/`, `web/demo_scans/` (directories)

- [ ] **Add FastAPI deps to pyproject.toml**

Replace the `dependencies` list:
```toml
dependencies = [
    "pandas",
    "numpy",
    "Pillow",
    "fastapi",
    "uvicorn[standard]",
    "python-multipart",
]
```

- [ ] **Create directories**

```bash
mkdir -p web/static web/demo_scans
```

- [ ] **Install dependencies**

```bash
uv sync
```

Expected: resolves without errors, `fastapi` and `uvicorn` appear in lock file.

- [ ] **Commit**

```bash
git add pyproject.toml
git -c commit.gpgsign=false commit -m "feat: add fastapi/uvicorn deps for web app"
```

---

## Task 2: Classifier Stub

**Files:**
- Create: `web/classifier.py`

The real classifier will replace this stub. The stub returns seeded-random probabilities so the same image always produces the same result during testing.

- [ ] **Create `web/classifier.py`**

```python
import numpy as np

CLASSES = ["Diabetes", "MS", "Glaucoma", "DryEye", "Healthy"]


def classify(image_bytes: bytes) -> dict[str, float]:
    """
    TODO: Replace with actual model inference.
    Accepts raw image bytes. Returns {class: probability} where each value is 0.0–1.0.
    Probabilities need not sum to 1.
    """
    seed = int.from_bytes(image_bytes[:8], "little") % (2**32)
    rng = np.random.default_rng(seed)
    raw = rng.exponential(scale=1.0, size=len(CLASSES))
    probs = (raw / raw.sum()).tolist()
    return dict(zip(CLASSES, probs))
```

- [ ] **Smoke-test the stub**

```bash
cd web
python -c "
from classifier import classify
r = classify(b'hello world test bytes')
print(r)
assert set(r.keys()) == {'Diabetes','MS','Glaucoma','DryEye','Healthy'}
assert all(0 <= v <= 1 for v in r.values())
print('OK')
"
```

Expected output: dict with 5 keys all between 0 and 1, then `OK`.

- [ ] **Commit**

```bash
git add web/classifier.py
git -c commit.gpgsign=false commit -m "feat: add classifier stub"
```

---

## Task 3: FastAPI Backend

**Files:**
- Create: `web/main.py`

- [ ] **Create `web/main.py`**

```python
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from classifier import classify

BASE = Path(__file__).parent
STATIC_DIR = BASE / "static"
DEMO_DIR = BASE / "demo_scans"

app = FastAPI(title="TearScan")


@app.post("/classify")
async def classify_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        probabilities = classify(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"filename": file.filename, "probabilities": probabilities}


@app.get("/api/demo-scans")
def list_demo_scans():
    if not DEMO_DIR.exists():
        return {"filenames": []}
    exts = {".bmp", ".png", ".tiff", ".tif"}
    files = sorted(f.name for f in DEMO_DIR.iterdir() if f.suffix.lower() in exts)
    return {"filenames": files}


if DEMO_DIR.exists():
    app.mount("/demo-scans", StaticFiles(directory=str(DEMO_DIR)), name="demo-scans")

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
```

- [ ] **Create a minimal `web/static/index.html` to test the server starts**

```html
<!DOCTYPE html><html><body><h1>TearScan</h1></body></html>
```

- [ ] **Start the server and test the API**

```bash
cd web && uvicorn main:app --reload &
sleep 2

# Test /classify with a sample BMP
curl -s -X POST http://localhost:8000/classify \
  -F "file=@../TRAIN_SET/Diabetes/37_DM.010_1.bmp" | python -m json.tool

# Test /api/demo-scans
curl -s http://localhost:8000/api/demo-scans

# Stop server
kill %1
```

Expected from `/classify`: JSON with `filename` and `probabilities` keys containing 5 disease values.
Expected from `/api/demo-scans`: `{"filenames": []}` (demo_scans is empty so far).

- [ ] **Commit**

```bash
git add web/main.py web/static/index.html
git -c commit.gpgsign=false commit -m "feat: add fastapi backend with classify and demo-scans endpoints"
```

---

## Task 4: HTML Shell

**Files:**
- Modify: `web/static/index.html`

- [ ] **Replace `web/static/index.html` with the full shell**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TearScan — Diagnostic Analysis System</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>

<nav>
  <span class="logo-wordmark">TearScan</span>
  <div class="logo-sep"></div>
  <span class="logo-sub">Diagnostic Analysis System</span>
  <div class="nav-actions">
    <button id="demo-btn" class="btn-secondary no-print">Load Demo Scans</button>
    <button id="export-btn" class="btn-secondary no-print">Export PDF</button>
  </div>
</nav>

<main>
  <div class="page-header no-print">
    <h1>Tear Sample Analysis</h1>
    <p class="subtitle">AFM crystallography pattern classification for disease biomarker detection</p>
  </div>

  <div id="upload-zone" class="upload-zone no-print">
    <input type="file" id="file-input" accept=".bmp,.png,.tiff,.tif" multiple hidden>
    <div class="upload-icon">⬆</div>
    <div class="upload-title">Drop scan images here or click to select</div>
    <div class="upload-hint">BMP · PNG · TIFF · Multiple files accepted</div>
    <div id="file-chips" class="queued-files"></div>
    <button id="analyse-btn" class="btn-primary">Analyse Scans</button>
  </div>

  <div id="status-bar" class="status-bar no-print" hidden>
    <div class="status-top">
      <span id="status-text" class="status-text"></span>
      <span id="status-count" class="status-count"></span>
    </div>
    <div class="progress-track">
      <div id="progress-fill" class="progress-fill" style="width:0%"></div>
    </div>
  </div>

  <div id="results-section" hidden>
    <div class="summary-strip no-print">
      <div class="summary-stat">
        <div id="count-total" class="stat-num num-neutral">0</div>
        <div class="stat-label">Total</div>
      </div>
      <div class="summary-stat">
        <div id="count-high" class="stat-num num-high">0</div>
        <div class="stat-label">High Risk</div>
      </div>
      <div class="summary-stat">
        <div id="count-warning" class="stat-num num-warn">0</div>
        <div class="stat-label">Warning</div>
      </div>
      <div class="summary-stat">
        <div id="count-safe" class="stat-num num-safe">0</div>
        <div class="stat-label">Clear</div>
      </div>
    </div>

    <div class="filter-bar no-print">
      <button class="filter-btn active" data-filter="all">All</button>
      <button class="filter-btn" data-filter="high">High Risk</button>
      <button class="filter-btn" data-filter="warning">Warning</button>
      <button class="filter-btn" data-filter="safe">Clear</button>
    </div>

    <div id="results-grid" class="results-grid"></div>
  </div>
</main>

<footer class="site-footer">
  Results are AI-assisted and intended for research use only. Not a clinical diagnosis.
</footer>

<div id="lightbox" class="lightbox" hidden>
  <div class="lightbox-inner">
    <button id="lightbox-close" class="lightbox-close">✕</button>
    <img id="lightbox-img" src="" alt="">
    <div id="lightbox-caption" class="lightbox-caption"></div>
  </div>
</div>

<script src="app.js"></script>
</body>
</html>
```

- [ ] **Commit**

```bash
git add web/static/index.html
git -c commit.gpgsign=false commit -m "feat: add html shell"
```

---

## Task 5: Styles

**Files:**
- Create: `web/static/styles.css`

- [ ] **Create `web/static/styles.css`**

```css
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: #f0ebe2;
  color: #1a1410;
  font-family: 'Segoe UI', system-ui, sans-serif;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* ── Nav ── */
nav {
  background: #fff;
  border-bottom: 1px solid #d8d0c4;
  padding: 18px 40px;
  display: flex;
  align-items: center;
  gap: 16px;
}
.logo-wordmark {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 22px;
  font-weight: 400;
  letter-spacing: 4px;
  text-transform: uppercase;
  color: #1a1410;
}
.logo-sep { width: 1px; height: 22px; background: #d8d0c4; margin: 0 12px; }
.logo-sub { font-size: 13px; color: #6a6058; }
.nav-actions { margin-left: auto; display: flex; gap: 10px; }

/* ── Main ── */
main {
  max-width: 1200px;
  width: 100%;
  margin: 0 auto;
  padding: 48px 32px;
  flex: 1;
}

h1 {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 32px;
  font-weight: 400;
  letter-spacing: -0.3px;
  margin-bottom: 8px;
}
.page-header { margin-bottom: 36px; }
.subtitle { color: #6a6058; font-size: 15px; }

/* ── Buttons ── */
.btn-primary {
  background: #1a1410;
  color: #f0ebe2;
  border: none;
  border-radius: 8px;
  padding: 13px 34px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  letter-spacing: 1px;
  text-transform: uppercase;
  transition: background 0.2s;
}
.btn-primary:hover { background: #3a3028; }

.btn-secondary {
  background: transparent;
  color: #4a4440;
  border: 1px solid #d8d0c4;
  border-radius: 7px;
  padding: 8px 18px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: border-color 0.2s, color 0.2s;
}
.btn-secondary:hover { border-color: #3d6e6a; color: #3d6e6a; }

/* ── Upload zone ── */
.upload-zone {
  border: 2px dashed #c8bfb4;
  border-radius: 12px;
  padding: 36px;
  text-align: center;
  cursor: pointer;
  background: #fff;
  margin-bottom: 24px;
  transition: border-color 0.2s, background 0.2s;
}
.upload-zone:hover,
.upload-zone.dragover { border-color: #3d6e6a; background: #f7f3ec; }
.upload-icon { font-size: 26px; margin-bottom: 10px; opacity: 0.4; }
.upload-title { font-size: 17px; color: #2a2420; font-weight: 500; margin-bottom: 6px; }
.upload-hint { font-size: 13px; color: #9a9088; margin-bottom: 18px; }
.queued-files {
  display: flex;
  gap: 8px;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 18px;
}
.queued-file {
  background: #f0ebe2;
  border: 1px solid #d8d0c4;
  border-radius: 6px;
  padding: 5px 12px;
  font-size: 13px;
  color: #4a4440;
  font-weight: 500;
}

/* ── Status / progress ── */
.status-bar {
  background: #fff;
  border: 1px solid #d8d0c4;
  border-radius: 8px;
  padding: 14px 20px 12px;
  margin-bottom: 24px;
}
.status-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
  font-size: 14px;
}
.status-text { color: #4a4440; font-weight: 500; }
.status-count { color: #9a9088; }
.progress-track {
  background: #ece6de;
  border-radius: 4px;
  height: 8px;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: #3d6e6a;
  border-radius: 4px;
  transition: width 0.3s ease;
  position: relative;
  overflow: hidden;
}
.progress-fill::after {
  content: '';
  position: absolute;
  top: 0; left: -60%;
  width: 50%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
  animation: shimmer 1.4s ease-in-out infinite;
}
@keyframes shimmer { 0% { left: -60%; } 100% { left: 120%; } }

/* ── Summary strip ── */
.summary-strip {
  display: flex;
  border: 1px solid #d8d0c4;
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 20px;
  background: #fff;
}
.summary-stat {
  flex: 1;
  padding: 18px 24px;
  text-align: center;
  border-right: 1px solid #ece4d8;
}
.summary-stat:last-child { border-right: none; }
.stat-num {
  font-family: Georgia, serif;
  font-size: 36px;
  font-weight: 400;
  letter-spacing: -1px;
}
.stat-label {
  font-size: 13px;
  color: #6a6058;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-top: 4px;
  font-weight: 500;
}
.num-neutral { color: #3d6e6a; }
.num-high    { color: #b82c2c; }
.num-warn    { color: #b06a10; }
.num-safe    { color: #2a7048; }

/* ── Filter bar ── */
.filter-bar { display: flex; gap: 8px; margin-bottom: 20px; }
.filter-btn {
  background: #fff;
  border: 1px solid #d8d0c4;
  border-radius: 7px;
  padding: 8px 18px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  color: #4a4440;
  transition: all 0.15s;
}
.filter-btn:hover { border-color: #3d6e6a; color: #3d6e6a; }
.filter-btn.active { background: #1a1410; color: #f0ebe2; border-color: #1a1410; }

/* ── Results grid ── */
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 16px;
}

/* ── Card ── */
.result-card {
  background: #fff;
  border: 1px solid #d8d0c4;
  border-radius: 12px;
  overflow: hidden;
  transition: box-shadow 0.2s;
}
.result-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.10); }

.card-image-wrap {
  position: relative;
  width: 100%;
  aspect-ratio: 1 / 1;
  background: #e8e0d4;
  overflow: hidden;
  border-bottom: 1px solid #d8d0c4;
  cursor: pointer;
}
.card-image-wrap img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.risk-overlay {
  position: absolute;
  top: 8px; right: 8px;
  padding: 4px 10px;
  border-radius: 5px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.risk-high { background: rgba(255,255,255,0.92); color: #b82c2c; border: 1.5px solid rgba(184,44,44,0.3); }
.risk-warn { background: rgba(255,255,255,0.92); color: #b06a10; border: 1.5px solid rgba(176,106,16,0.3); }
.risk-safe { background: rgba(255,255,255,0.92); color: #2a7048; border: 1.5px solid rgba(42,112,72,0.3); }

.card-meta { padding: 10px 12px 8px; border-bottom: 1px solid #ece6de; }
.card-filename {
  font-size: 13px;
  color: #1a1410;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  display: block;
}

.card-body { padding: 12px; }
.arcs-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px 4px;
}
.arc-item { display: flex; flex-direction: column; align-items: center; gap: 3px; }
.arc-item svg { width: 100%; max-width: 64px; height: auto; }
.arc-label {
  font-size: 13px;
  color: #2a2420;
  font-weight: 600;
  text-align: center;
  line-height: 1.2;
}

/* ── Footer ── */
.site-footer {
  text-align: center;
  padding: 20px 32px;
  font-size: 12px;
  color: #9a9088;
  border-top: 1px solid #d8d0c4;
  background: #faf6f0;
}

/* ── Lightbox ── */
.lightbox {
  position: fixed;
  inset: 0;
  background: rgba(20,16,12,0.88);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.lightbox[hidden] { display: none; }
.lightbox-inner {
  position: relative;
  max-width: 90vw;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}
.lightbox-inner img {
  max-width: 100%;
  max-height: 80vh;
  object-fit: contain;
  border-radius: 4px;
}
.lightbox-close {
  position: absolute;
  top: -40px; right: 0;
  background: transparent;
  border: 1px solid rgba(255,255,255,0.3);
  color: #fff;
  border-radius: 50%;
  width: 32px; height: 32px;
  font-size: 14px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}
.lightbox-caption { color: rgba(255,255,255,0.7); font-size: 13px; }

/* ── Print / PDF ── */
@media print {
  .no-print { display: none !important; }
  body { background: #fff; }
  main { padding: 0; max-width: 100%; }
  .results-grid { grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
  .result-card { break-inside: avoid; border: 1px solid #ccc; box-shadow: none; }
  .site-footer { border-top: 1px solid #ccc; }
  nav { border-bottom: 1px solid #ccc; padding: 12px 20px; }
  #results-section { display: block !important; }
}
```

- [ ] **Verify server shows styled page**

```bash
cd web && uvicorn main:app &
sleep 2
curl -s http://localhost:8000/ | grep -q "TearScan" && echo "OK"
kill %1
```

- [ ] **Commit**

```bash
git add web/static/styles.css
git -c commit.gpgsign=false commit -m "feat: add styles"
```

---

## Task 6: JavaScript

**Files:**
- Create: `web/static/app.js`

- [ ] **Create `web/static/app.js`**

```javascript
'use strict';

const DISEASES = ['Diabetes', 'MS', 'Glaucoma', 'DryEye', 'Healthy'];
const DISEASE_LABELS = {
  Diabetes: 'Diabetes', MS: 'MS', Glaucoma: 'Glaucoma',
  DryEye: 'Dry Eye', Healthy: 'Healthy',
};
const CIRC = 87.96; // 2π × r=14

let results = [];
let activeFilter = 'all';
let classifying = false;
let queuedFiles = [];

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
const BADGE_LABEL = { high: 'High Risk', warning: 'Warning', safe: 'Clear' };

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
  classifying = true;
  results = [];

  const statusBar     = document.getElementById('status-bar');
  const resultsSection = document.getElementById('results-section');
  const progressFill  = document.getElementById('progress-fill');
  const statusText    = document.getElementById('status-text');
  const statusCount   = document.getElementById('status-count');

  statusBar.hidden = false;
  resultsSection.hidden = false;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    statusText.textContent = `Classifying ${file.name}\u2026`;
    statusCount.textContent = `${i + 1} of ${files.length}`;
    progressFill.style.width = `${(i / files.length) * 100}%`;

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
    }
  }

  progressFill.style.width = '100%';
  statusText.textContent = 'Classification complete';
  statusCount.textContent = `${files.length} of ${files.length}`;
  classifying = false;
}

// ── File queue ──────────────────────────────────────────────
function updateChips() {
  document.getElementById('file-chips').innerHTML =
    queuedFiles.map(f => `<div class="queued-file">${f.name}</div>`).join('');
}

function addFiles(list) {
  queuedFiles = [...queuedFiles, ...Array.from(list)];
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

  uploadZone.addEventListener('click', () => fileInput.click());
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
    .addEventListener('click', () => runClassification(queuedFiles));
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
});
```

- [ ] **Commit**

```bash
git add web/static/app.js
git -c commit.gpgsign=false commit -m "feat: add frontend javascript"
```

---

## Task 7: Demo Scans

**Files:**
- Populate: `web/demo_scans/`
- Modify: `.gitignore`

- [ ] **Copy one BMP from each class into demo_scans/**

```bash
cp "TRAIN_SET/Diabetes/37_DM.010_1.bmp"               web/demo_scans/demo_diabetes.bmp
cp "TRAIN_SET/PGOV_Glaukom/21_LV_PGOV+SII.000_1.bmp"  web/demo_scans/demo_glaucoma.bmp
cp "TRAIN_SET/SklerózaMultiplex/1-SM-LM-18.000_1.bmp"  web/demo_scans/demo_ms.bmp
cp "TRAIN_SET/SucheOko/29_PM_suche_oko.000_1.bmp"      web/demo_scans/demo_dry_eye.bmp
cp "TRAIN_SET/ZdraviLudia/1L_M.000_1.bmp"              web/demo_scans/demo_healthy.bmp
```

- [ ] **Add demo_scans to .gitignore** (BMPs are large, same as TRAIN_SET)

Append to `.gitignore`:
```
web/demo_scans/
```

- [ ] **Verify the API lists them**

```bash
cd web && uvicorn main:app &
sleep 2
curl -s http://localhost:8000/api/demo-scans
kill %1
```

Expected: `{"filenames":["demo_diabetes.bmp","demo_dry_eye.bmp","demo_glaucoma.bmp","demo_healthy.bmp","demo_ms.bmp"]}`

- [ ] **Commit**

```bash
git add .gitignore
git -c commit.gpgsign=false commit -m "feat: add demo scans gitignore entry"
```

---

## Task 8: End-to-End Smoke Test

- [ ] **Start the server**

```bash
cd web && uvicorn main:app --reload
```

Open http://localhost:8000 in a browser.

- [ ] **Upload test**

Drag 3 BMP files from `TRAIN_SET/` onto the upload zone. Verify all filenames appear as chips (no "+N more" truncation).

- [ ] **Classification test**

Click "Analyse Scans". Verify:
- Progress bar fills with shimmer animation
- Status text shows current filename and `N of M` counter
- Cards appear one by one, sorted High Risk → Warning → Clear
- Each card has a square image, risk badge, and 5 arc gauges

- [ ] **Lightbox test**

Click any card image. Verify full-size overlay appears. Press Escape — overlay closes. Click the overlay background — closes. Click ✕ button — closes.

- [ ] **Filter test**

Click "High Risk" filter button — only high-risk cards shown, button turns dark. Click "All" — all cards return.

- [ ] **Demo mode test**

Refresh the page. Click "Load Demo Scans" in the nav. Verify 5 demo scans are classified automatically.

- [ ] **PDF export test**

Click "Export PDF". Verify browser print dialog opens with only results visible (no upload zone, no nav buttons, no filter bar).

- [ ] **Disclaimer test**

Scroll to bottom — confirm footer text reads: *"Results are AI-assisted and intended for research use only. Not a clinical diagnosis."*

- [ ] **Final commit**

```bash
git -c commit.gpgsign=false commit --allow-empty -m "feat: tearscan web app complete"
```
