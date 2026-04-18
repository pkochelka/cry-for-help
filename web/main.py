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
