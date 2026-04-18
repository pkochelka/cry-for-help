import numpy as np

CLASSES = ["Diabetes", "MS", "Glaucoma", "DryEye", "Healthy"]


def classify(image_bytes: bytes) -> dict[str, float]:
    """
    TODO: Replace with actual model inference.
    Accepts raw image bytes. Returns {class: probability} where each value is 0.0–1.0.
    Probabilities need not sum to 1.
    """
    seed = hash(image_bytes) % (2**32)  # Ensure seed is a non-negative integer
    print(seed)
    rng = np.random.default_rng(seed)
    raw = rng.exponential(scale=1.0, size=len(CLASSES))
    probs = (raw / raw.sum()).tolist()
    return dict(zip(CLASSES, probs))
