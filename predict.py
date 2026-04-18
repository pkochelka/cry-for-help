"""
Inference for the trained tear microscopy classifier.

Requires:
  - model.joblib          (sklearn pipeline + LabelEncoder)
  - model_meta.json       (backbone id, classes, TTA settings, ...)
  - augmentation.py       (same transforms used at training time)

Usage:
  python predict.py IMAGE_PATH SCALE_UM
  e.g.  python predict.py sample.png 50
"""

import sys 
import json
import joblib
import numpy as np
import torch
import timm
from sklearn.preprocessing import normalize
from augmentation import build_eval_transform, build_train_transform
from bmp_data_processing.scripts.bmp_to_pd import preprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_bundle(model_path="model.joblib", meta_path="model_meta.json"):
    bundle = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    backbone = timm.create_model(
        meta["backbone_timm_id"], pretrained=True, num_classes=0, global_pool="avg"
    ).eval().to(DEVICE)
    return bundle["model"], bundle["label_encoder"], backbone, meta


def extract_feature(img_arr, backbone, meta):
    """Mean of 1 eval view + n_tta augmented views, L2-normalized, + scale placeholder."""
    eval_tfm  = build_eval_transform()
    train_tfm = build_train_transform()
    n_tta = meta.get("n_tta", 8)
    views = [eval_tfm(img_arr)] + [train_tfm(img_arr) for _ in range(n_tta)]
    with torch.no_grad():
        x = torch.stack(views).to(DEVICE)
        f = backbone(x).cpu().numpy().mean(axis=0, keepdims=True)
    if meta.get("feature_l2_normalize", True):
        f = normalize(f, norm="l2")
    return f


def predict(image_path,
            model_path="model.joblib", meta_path="model_meta.json"):
    clf, le, backbone, meta = load_bundle(model_path, meta_path)
    result = preprocess(image_path, None)
    f = extract_feature(result["pixels"], backbone, meta)
    if meta.get("use_scale_feature", True):
        f = np.hstack([f, np.array([[result["scale"]]])])

    probs = clf.predict_proba(f)[0]
    idx   = int(np.argmax(probs))
    label = le.classes_[idx]

    print(f"\nImage: {image_path}   scale: {result['scale']}")
    print(f"Prediction: {label}\n")
    print("Per-class probabilities:")
    for cls, p in sorted(zip(le.classes_, probs), key=lambda t: -t[1]):
        print(f"  {cls:20s} {p:.3f}")
    return label, dict(zip(le.classes_.tolist(),
                           [float(p) for p in probs]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py IMAGE_PATH SCALE_UM")
        sys.exit(1)
    predict(sys.argv[1])