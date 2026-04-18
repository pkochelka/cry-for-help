from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.classifier import classify, get_model_status, start_model_warmup

BASE = Path(__file__).parent
STATIC_DIR = BASE / "static"
DEMO_DIR = BASE / "demo_scans"

app = FastAPI(title="CryForHelp")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC_DIR / "favicon.ico")
app = FastAPI(title="TearScan")
start_model_warmup()


@app.post("/classify")
async def classify_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".bmp"):
        raise HTTPException(400, "Please upload a .bmp file")
    try:
        result = classify(await file.read())
        return {"filename": file.filename, **result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


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
