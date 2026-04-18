# TearScan Web App — Design Spec

## Context

Hackathon demo tool for classifying dried tear drop AFM scan images. An external Python classifier service accepts images and returns per-disease probabilities. The web app is the front-end + API layer that makes this accessible and visually compelling for a live audience and clinicians.

---

## Architecture

```
Browser (Vanilla HTML/CSS/JS)
        │  multipart POST /classify
        ▼
FastAPI backend (Python)
        │  calls
        ▼
Classifier service (existing Python, imported or subprocess)
```

**Frontend:** Single-page app — one `index.html` + `app.js` + `styles.css`. No build tooling. Served by FastAPI as static files.

**Backend:** FastAPI app with two routes:
- `POST /classify` — accepts one image file, returns JSON `{ filename, probabilities: { Diabetes, MS, Glaucoma, DryEye, Healthy } }`
- `GET /` — serves the frontend

Images are sent one at a time (frontend fires them in parallel or sequence). No persistence — session-only.

---

## Disease Categories & Thresholds

Five classes from the dataset:
- Diabetes, Multiple Sclerosis (MS), Glaucoma, Dry Eye, Healthy (ZdraviLudia)

Risk thresholds applied **per disease per card**:
- **0–20%** → safe (green `#3a8858`)
- **20–60%** → warning (amber `#c07a18`)
- **60%+** → high risk (red `#b82c2c`)

Overall card badge = highest risk level across all diseases.

---

## Visual Design

Warm ivory light theme (`#f0ebe2` background, `#fff` cards). Serif wordmark (Georgia). Dark near-black text for contrast. Muted teal (`#3d6e6a`) as accent/action color.

**Card layout:**
- Square `aspect-ratio: 1/1` image preview at top (actual scan BMP rendered via `<img>`)
- Risk badge overlaid on image (High Risk / Warning / Clear)
- Filename below image
- 3-column grid of SVG arc gauges (one per disease): arc fill + percentage inside + label below
- Top-predicted disease name displayed bold in a highlighted row above the arc grid

**Page layout:**
- Nav: wordmark + subtitle
- Upload zone: drag & drop + click, all queued filenames listed as chips
- Progress bar with shimmer animation (updates as each image is classified)
- Summary strip: Total / High Risk / Warning / Clear counts
- Filter toggle bar: All | High Risk | Warning | Clear
- Results grid: `repeat(auto-fill, minmax(240px, 1fr))`, sorted high → warning → clear
- Footer: confidence disclaimer

---

## Features

### Core
- **Multi-image upload** — drag & drop or file picker, no limit, all filenames listed
- **Sequential classification** — images sent to `/classify` one at a time; cards appear as results arrive
- **Progress bar** — teal fill + shimmer runner, `N of M` counter
- **Sort by risk** — results grid always sorted: High Risk first, then Warning, then Clear

### Results Display
- **Arc gauges** — SVG circles per disease, color-coded by threshold
- **Risk badge** — overlaid on scan image

### Interactivity
- **Filter by risk level** — toggle buttons (All / High Risk / Warning / Clear); hides non-matching cards
- **Image lightbox** — click any scan image to open full-size overlay; close with Escape or click outside

### Demo Mode
- Button in nav/upload area: "Load Demo Scans"
- Loads 3–5 pre-bundled sample BMP files (one per risk category) and runs classification automatically
- Useful when live uploading isn't practical

### Export
- **PDF export** — "Export PDF" button triggers `window.print()` with a print stylesheet that hides upload zone, progress bar, buttons and renders the results grid cleanly across pages

### Footer
- Single-line disclaimer: *"Results are AI-assisted and intended for research use only. Not a clinical diagnosis."*

---

## File Structure

```
hack-kosice-tears/
├── web/
│   ├── main.py          # FastAPI app
│   ├── classifier.py    # thin wrapper around existing classifier
│   ├── static/
│   │   ├── index.html
│   │   ├── app.js
│   │   └── styles.css
│   └── demo_scans/      # 3–5 sample BMP files (one per class)
└── ...existing project files...
```

---

## API Contract

```
POST /classify
Content-Type: multipart/form-data
Body: file=<image bytes>

Response 200:
{
  "filename": "scan_001.bmp",
  "probabilities": {
    "Diabetes": 0.82,
    "MS": 0.41,
    "Glaucoma": 0.12,
    "DryEye": 0.08,
    "Healthy": 0.03
  }
}

Response 422: { "error": "..." }
```

---

## Verification

1. `cd web && uvicorn main:app --reload` — server starts, static files served at `/`
2. Open browser, drag 3+ BMP files → all filenames appear as chips
3. Click Analyse → progress bar fills, cards appear in order of completion, sorted by risk
4. Click a scan image → lightbox opens full-size; Escape closes it
5. Click "Load Demo Scans" → pre-bundled scans classify automatically
6. Toggle "High Risk" filter → only high-risk cards visible
7. Click "Export PDF" → browser print dialog, results render cleanly without UI chrome
8. Confirm disclaimer visible in footer
