import os
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from bmp_data_processing.scripts.bmp_to_pd import preprocess
from predict import extract_feature, load_bundle

CLASSES = ["Diabetes", "MS", "Glaucoma", "DryEye", "Healthy"]

_LABEL_ALIASES = {
    "Diabetes": "Diabetes",
    "MS": "MS",
    "SklerozaMultiplex": "MS",
    "SklerózaMultiplex": "MS",
    "PGOV_Glaukom": "Glaucoma",
    "Glaucoma": "Glaucoma",
    "SucheOko": "DryEye",
    "DryEye": "DryEye",
    "ZdraviLudia": "Healthy",
    "Healthy": "Healthy",
}


def _normalize_probs(classes, probs) -> dict:
    out = {c: 0.0 for c in CLASSES}
    for cls, p in zip(classes, probs):
        key = _LABEL_ALIASES.get(cls)
        if key:
            out[key] += float(p)
    return out

from classifier import classify, get_model_status, start_model_warmup

BASE = Path(__file__).parent
STATIC_DIR = BASE / "static"
DEMO_DIR = BASE / "demo_scans"

app = FastAPI(title="CryForHelp")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load model once at startup
clf, le, backbone, meta = load_bundle()


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC_DIR / "favicon.ico")
app = FastAPI(title="TearScan")
start_model_warmup()


@app.post("/classify")
async def classify_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".bmp"):
        raise HTTPException(400, "Please upload a .bmp file")

    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = preprocess(tmp_path, None)
        f = extract_feature(result["pixels"], backbone, meta)
        if meta.get("use_scale_feature", True):
            f = np.hstack([f, np.array([[result["scale"]]])])

        probs = clf.predict_proba(f)[0]
        idx = int(np.argmax(probs))
        raw_label = le.classes_[idx]
        norm = _normalize_probs(le.classes_, probs)
        return {
            "filename": file.filename,
            "label": _LABEL_ALIASES.get(raw_label, raw_label),
            "scale": float(result["scale"]),
            "probabilities": norm,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        os.unlink(tmp_path)


@app.get("/api/demo-scans")
def list_demo_scans():
    if not DEMO_DIR.exists():
        return {"filenames": []}
    exts = {".bmp", ".png", ".tiff", ".tif"}
    files = sorted(f.name for f in DEMO_DIR.iterdir() if f.suffix.lower() in exts)
    return {"filenames": files}


@app.get("/api/model-status")
def model_status():
    return get_model_status()


if DEMO_DIR.exists():
    app.mount("/demo-scans", StaticFiles(directory=str(DEMO_DIR)), name="demo-scans")

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")