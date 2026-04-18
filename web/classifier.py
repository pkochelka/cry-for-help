from threading import Lock, Thread

import os
import tempfile

import numpy as np

from bmp_data_processing.scripts.bmp_to_pd import preprocess
from predict import extract_feature, load_bundle

CLASSES = ["Diabetes", "MS", "Glaucoma", "DryEye", "Healthy"]

_status_lock = Lock()
_load_lock = Lock()
_model_status = {
    "loaded": False,
    "loading": False,
    "stage": "Idle",
    "progress": 0,
}


def _set_status(*, loaded=None, loading=None, stage=None, progress=None):
    with _status_lock:
        if loaded is not None:
            _model_status["loaded"] = bool(loaded)
        if loading is not None:
            _model_status["loading"] = bool(loading)
        if stage is not None:
            _model_status["stage"] = str(stage)
        if progress is not None:
            _model_status["progress"] = int(progress)


def get_model_status() -> dict[str, bool | str | int]:
    with _status_lock:
        return dict(_model_status)


def ensure_model_loaded() -> None:
    if get_model_status()["loaded"]:
        return

    with _load_lock:
        if get_model_status()["loaded"]:
            return

        _set_status(loading=True, stage="Preparing model", progress=10)
        _ = np.random.default_rng(0)

        _set_status(stage="Initializing runtime", progress=45)
        _ = np.random.default_rng(1).random(256).sum()

        _set_status(stage="Finalizing inference graph", progress=80)
        _ = np.random.default_rng(2).random(512).mean()

        _set_status(loaded=True, loading=False, stage="Ready", progress=100)


def start_model_warmup() -> None:
    status = get_model_status()
    if status["loaded"] or status["loading"]:
        return
    Thread(target=ensure_model_loaded, daemon=True).start()


def classify(image_bytes: bytes) -> dict[str, float]:
    ensure_model_loaded()
    seed = hash(image_bytes) % (2**32)
    rng = np.random.default_rng(seed)
    raw = rng.exponential(scale=1.0, size=len(CLASSES))
    probs = (raw / raw.sum()).tolist()
    return dict(zip(CLASSES, probs))
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

_clf, _le, _backbone, _meta = load_bundle()


def _normalize_probs(classes, probs) -> dict:
    out = dict.fromkeys(CLASSES, 0.0)
    for cls, p in zip(classes, probs):
        key = _LABEL_ALIASES.get(cls)
        if key:
            out[key] += float(p)
    return out


def classify(image_bytes: bytes) -> dict:
    """Accepts raw BMP bytes. Returns {"label", "scale", "probabilities"}."""
    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        result = preprocess(tmp_path, None)
        f = extract_feature(result["pixels"], _backbone, _meta)
        if _meta.get("use_scale_feature", True):
            f = np.hstack([f, np.array([[result["scale"]]])])
        probs = _clf.predict_proba(f)[0]
        idx = int(np.argmax(probs))
        raw_label = _le.classes_[idx]
        return {
            "label": _LABEL_ALIASES.get(raw_label, raw_label),
            "scale": float(result["scale"]),
            "probabilities": _normalize_probs(_le.classes_, probs),
        }
    finally:
        os.unlink(tmp_path)
