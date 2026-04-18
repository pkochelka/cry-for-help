from threading import Lock, Thread

import numpy as np

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
