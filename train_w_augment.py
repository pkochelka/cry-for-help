"""
Tear microscopy classification (5 classes, ~300 imbalanced samples).

- L2-normalized backbone features + raw `scale` feature.
- TTA on val, heavy train-time aug (N_TRAIN_AUG views per image).
- ConvNeXt-V2 + DINOv2 backbones, LogReg/SVM classifiers, probability-averaging ensemble.
- Saves model.joblib + model_meta.json for inference.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import timm
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from augmentation import build_eval_transform, build_train_transform

CACHE_DIR = "features_cache_v2"
os.makedirs(CACHE_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_TRAIN_AUG, N_TTA = 6, 8

BACKBONES = {
    "dinov2_base":     {"timm_id": "vit_base_patch14_dinov2.lvd142m",     "img_size": 518},
    "swinv2_base":     {"timm_id": "swinv2_base_window16_256.ms_in1k",    "img_size": 256},
    "convnextv2_base": {"timm_id": "convnextv2_base.fcmae_ft_in22k_in1k", "img_size": 256},
}


# ---------- Feature extraction ----------
def build_backbone(name):
    cfg = BACKBONES[name]
    return timm.create_model(cfg["timm_id"], pretrained=True, num_classes=0,
                             img_size=cfg["img_size"]).eval().to(DEVICE)


def build_transforms(img_size):
    return build_train_transform(size=img_size), build_eval_transform(size=img_size)


def _finalize(feats, scales):
    X = normalize(np.vstack(feats), norm="l2")
    return np.hstack([X, np.array(scales).reshape(-1, 1)])


def _extract(df, model, mode, train_tfm, eval_tfm):
    """mode: 'train' -> N_TRAIN_AUG augmented rows per image;
             'tta'   -> 1 row per image, mean of eval + N_TTA augmented views."""
    feats, labels, scales = [], [], []
    with torch.no_grad():
        for i, row in df.iterrows():
            if i % 25 == 0:
                print(f"  {i}/{len(df)}")
            img = row["pixels"].astype(np.uint8)
            if mode == "train":
                for _ in range(N_TRAIN_AUG):
                    x = train_tfm(img).unsqueeze(0).to(DEVICE)
                    feats.append(model(x).cpu().numpy().squeeze())
                    labels.append(row["label"]); scales.append(row["scale"])
            else:
                views = [eval_tfm(img)] + [train_tfm(img) for _ in range(N_TTA)]
                x = torch.stack(views).to(DEVICE)
                feats.append(model(x).cpu().numpy().mean(axis=0))
                labels.append(row["label"]); scales.append(row["scale"])
    return _finalize(feats, scales), np.array(labels)


def load_or_extract(bname, tag, df, mode):
    path = os.path.join(CACHE_DIR, f"{bname}_{tag}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        print(f"Loaded cached {bname}/{tag}")
        return d["X"], d["y"]
    print(f"Extracting {bname}/{tag}...")
    m = build_backbone(bname)
    train_tfm, eval_tfm = build_transforms(BACKBONES[bname]["img_size"])
    X, y = _extract(df, m, mode, train_tfm, eval_tfm)
    np.savez(path, X=X, y=y)
    del m
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return X, y


# ---------- Classifiers ----------
def get_classifiers():
    return {
        "LogReg_L2":  LogisticRegression(C=5.0, max_iter=3000, class_weight="balanced",
                                         solver="lbfgs", random_state=0),
        "LogReg_L1":  LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced",
                                         solver="saga", random_state=1),
        "SVM_linear": SVC(C=1.0, kernel="linear", class_weight="balanced",
                          probability=True, random_state=2),
        "SVM_rbf":    SVC(C=5.0, kernel="rbf", gamma="scale", class_weight="balanced",
                          probability=True, random_state=3),
    }


def make_pipe(clf_name):
    return Pipeline([("scaler", StandardScaler()), ("clf", get_classifiers()[clf_name])])


def benchmark(X, y):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    rows = []
    for name in get_classifiers():
        s = cross_val_score(make_pipe(name), X, y, cv=cv,
                            scoring="f1_weighted", n_jobs=-1)
        rows.append((name, s.mean(), s.std()))
        print(f"    {name:12s}  F1w = {s.mean():.3f} ± {s.std():.3f}")
    rows.sort(key=lambda r: -r[1])
    return pd.DataFrame(rows, columns=["clf", "f1_mean", "f1_std"])


def plot_confusion(y_true, y_pred, classes, title, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=classes).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close(fig)


class ProbAvgEnsemble:
    def __init__(self, pipelines, label_order):
        self.pipelines, self.label_order = pipelines, label_order

    def predict_proba(self, Xs):
        return np.mean([p.predict_proba(Xs[b]) for b, p in self.pipelines.items()], axis=0)

    def predict(self, Xs):
        return np.argmax(self.predict_proba(Xs), axis=1)


# ---------- Main ----------
if __name__ == "__main__":
    df = pd.read_pickle("out.pkl")
    train_df, val_df = train_test_split(df, test_size=0.2,
                                        stratify=df["label"], random_state=0)
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    le = LabelEncoder().fit(train_df["label"])

    # --- Per-backbone: extract, benchmark, pick best single, fit, evaluate ---
    per_bb = {}
    for bname in BACKBONES:
        print(f"\n{'='*60}\nBackbone: {bname}\n{'='*60}")
        try:
            X_tr, y_tr_raw = load_or_extract(bname, "train",   train_df, "train")
            X_va, y_va_raw = load_or_extract(bname, "val_tta", val_df,   "tta")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {e}")
            continue

        y_tr, y_va = le.transform(y_tr_raw), le.transform(y_va_raw)
        print(f"  Train: {X_tr.shape}, Val: {X_va.shape}")

        best = benchmark(X_tr, y_tr).iloc[0]["clf"]
        pipe = make_pipe(best); pipe.fit(X_tr, y_tr)
        val_f1 = f1_score(y_va, pipe.predict(X_va), average="weighted", zero_division=0)
        print(f"  --> Best single: {best}, Val F1w = {val_f1:.3f}")
        per_bb[bname] = (best, X_tr, y_tr, X_va, y_va, val_f1)

    if not per_bb:
        raise RuntimeError("No backbone succeeded.")

    # Best backbone by honest held-out F1
    best_bb = max(per_bb, key=lambda b: per_bb[b][5])
    best_clf_name, X_tr, y_tr, X_va, y_va, single_f1 = per_bb[best_bb]
    print(f"\nBest backbone by Val F1w: {best_bb} ({best_clf_name} = {single_f1:.3f})")

    # Fit the single-backbone winner
    final_model = make_pipe(best_clf_name); final_model.fit(X_tr, y_tr)
    final_name, final_f1 = best_clf_name, single_f1

    print(f"\nSingle-backbone: {best_bb} + {final_name}, Val F1w = {final_f1:.3f}")
    print(classification_report(y_va, final_model.predict(X_va), target_names=le.classes_, zero_division=0))
    plot_confusion(y_va, final_model.predict(X_va), le.classes_,
                   f"{best_bb} + {final_name} (Val F1w={final_f1:.3f})",
                   f"confmat_{best_bb}_{final_name}.png")

    # --- Backbone ensemble via probability averaging (only if >=2 backbones) ---
    ensemble_meta = None
    if len(per_bb) >= 2:
        print(f"\n{'='*60}\nBACKBONE ENSEMBLE (prob averaging)\n{'='*60}")
        bb_names = list(per_bb)
        y_va_ens = per_bb[bb_names[0]][4]

        fitted = {}
        for b in bb_names:
            clf_b, X_tr_b, y_tr_b, *_ = per_bb[b]
            p = make_pipe(clf_b); p.fit(X_tr_b, y_tr_b)
            fitted[b] = p
        prob_va = np.mean([fitted[b].predict_proba(per_bb[b][3]) for b in bb_names], axis=0)
        f1_avg = f1_score(y_va_ens, np.argmax(prob_va, axis=1), average="weighted")
        print(f"  Prob averaging  Val F1w = {f1_avg:.3f}  |  Best single: {single_f1:.3f}")

        if f1_avg > final_f1:
            final_model = ProbAvgEnsemble(fitted, le.classes_)
            final_name, final_f1 = "ensemble_prob_avg", f1_avg
            y_pred_ens = final_model.predict({b: per_bb[b][3] for b in bb_names})
            ensemble_meta = {"strategy": "prob_avg", "backbones": bb_names}
            print(f"  --> Ensemble wins; switching to {final_name}")
            plot_confusion(y_va_ens, y_pred_ens, le.classes_,
                           f"{final_name} (Val F1w={final_f1:.3f})",
                           f"confmat_{final_name}.png")
        else:
            print("  --> Single-backbone model still wins; keeping it.")
    else:
        print("\nOnly one backbone available — skipping ensemble.")

    # --- Save ---
    joblib.dump({"model": final_model, "label_encoder": le}, "model.joblib")
    meta = {
        "backbone_name":        best_bb,
        "backbone_timm_id":     BACKBONES[best_bb]["timm_id"],
        "classifier":           final_name,
        "final_size":           BACKBONES[best_bb]["img_size"],
        "classes":              le.classes_.tolist(),
        "val_f1_weighted":      float(final_f1),
        "feature_l2_normalize": True,
        "scale_is_feature":     True,
        "n_tta":                N_TTA,
    }
    if ensemble_meta:
        meta["ensemble"] = ensemble_meta
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)