"""
Improved training for tear microscopy classification (5 classes, ~300 imbalanced samples).

Changes vs v1:
  - L2-normalize backbone features before concatenating `scale` feature.
  - Test-Time Augmentation (TTA) on val: mean of eval + N augmented feature views.
  - Adds DINOv2 as a candidate backbone alongside ConvNeXt-V2.
  - Adds RBF-SVM classifier.
  - Heavier training-time augmentation (N_TRAIN_AUG = 6).
  - Skips random_oversample() to avoid duplicate leak across CV folds;
    class_weight="balanced" + augmentation already handle imbalance.
  - Saves final pipeline + metadata (model.joblib, model_meta.json) for inference.
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
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from augmentation import build_eval_transform, build_train_transform

CACHE_DIR = "features_cache_v2"
os.makedirs(CACHE_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_TRAIN_AUG = 6   # augmented feature views per training image
N_TTA       = 8   # augmented views for TTA on val/test

train_tfm = build_train_transform()
eval_tfm  = build_eval_transform()

BACKBONES = {
    "convnextv2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
    "dinov2_vitb14":   "vit_base_patch14_dinov2.lvd142m",
}


# ---------- Data ----------
def load_data():
    return pd.read_pickle("out.pkl")

def stratified_split(df, test_size=0.2, random_state=0):
    tr, va = train_test_split(df, test_size=test_size,
                              stratify=df["label"], random_state=random_state)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


# ---------- Backbone ----------
def build_backbone(name):
    m = timm.create_model(BACKBONES[name], pretrained=True,
                          num_classes=0, global_pool="avg")
    return m.eval().to(DEVICE)


# ---------- Feature extraction ----------
def _finalize(feats, scales):
    X = np.vstack(feats)
    X = normalize(X, norm="l2")                                # L2-norm deep features
    X = np.hstack([X, np.array(scales).reshape(-1, 1)])        # append raw scale
    return X

def extract_train(df, model, n_aug=N_TRAIN_AUG):
    feats, labels, scales = [], [], []
    with torch.no_grad():
        for i, row in df.iterrows():
            if i % 25 == 0: print(f"  {i}/{len(df)}")
            img = row["pixels"].astype(np.uint8)
            for _ in range(n_aug):
                x = train_tfm(img).unsqueeze(0).to(DEVICE)
                feats.append(model(x).cpu().numpy().squeeze())
                labels.append(row["label"]); scales.append(row["scale"])
    return _finalize(feats, scales), np.array(labels)

def extract_tta(df, model, n_tta=N_TTA):
    """For each image: average feature over 1 eval view + n_tta augmented views."""
    feats, labels, scales = [], [], []
    with torch.no_grad():
        for i, row in df.iterrows():
            if i % 25 == 0: print(f"  {i}/{len(df)}")
            img = row["pixels"].astype(np.uint8)
            views = [eval_tfm(img)] + [train_tfm(img) for _ in range(n_tta)]
            x = torch.stack(views).to(DEVICE)
            f = model(x).cpu().numpy().mean(axis=0)
            feats.append(f); labels.append(row["label"]); scales.append(row["scale"])
    return _finalize(feats, scales), np.array(labels)


def load_or_extract(bname, tag, df, extractor):
    path = os.path.join(CACHE_DIR, f"{bname}_{tag}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        print(f"Loaded cached {bname}/{tag}")
        return d["X"], d["y"]
    print(f"Extracting {bname}/{tag}...")
    m = build_backbone(bname)
    X, y = extractor(df, m)
    np.savez(path, X=X, y=y)
    del m
    if DEVICE == "cuda": 
        torch.cuda.empty_cache()
    return X, y


# ---------- Classifiers ----------
def get_classifiers():
    return {
        "LogReg_L2":  LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced",
                                         solver="lbfgs", random_state=0),
        "LogReg_L1":  LogisticRegression(C=0.5,  max_iter=3000, class_weight="balanced",
                                         l1_ratio="l1", solver="saga", random_state=1),
        "SVM_linear": SVC(C=1.0, kernel="linear", class_weight="balanced",
                          probability=True, random_state=2),
        "SVM_rbf":    SVC(C=5.0, kernel="rbf", gamma="scale", class_weight="balanced",
                          probability=True, random_state=3),
    }

def benchmark(X, y):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    rows = []
    for name, clf in get_classifiers().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        s = cross_val_score(pipe, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        rows.append((name, s.mean(), s.std()))
        print(f"    {name:12s}  F1w = {s.mean():.3f} ± {s.std():.3f}")
    rows.sort(key=lambda r: -r[1])
    return pd.DataFrame(rows, columns=["clf", "f1_mean", "f1_std"])


def plot_confusion(y_true, y_pred, classes, title, out):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close(fig)


# ---------- Main ----------
if __name__ == "__main__":
    df = load_data()
    train_df, val_df = stratified_split(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    le = LabelEncoder().fit(train_df["label"])

    all_rows, per_bb = [], {}
    for bname in BACKBONES:
        print(f"\n{'='*60}\nBackbone: {bname}\n{'='*60}")
        try:
            X_tr, y_tr_raw = load_or_extract(bname, "train",   train_df, extract_train)
            X_va, y_va_raw = load_or_extract(bname, "val_tta", val_df,   extract_tta)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {e}"); continue

        y_tr, y_va = le.transform(y_tr_raw), le.transform(y_va_raw)
        print(f"  Train: {X_tr.shape}, Val: {X_va.shape}")

        res = benchmark(X_tr, y_tr)
        best = res.iloc[0]["clf"]
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", get_classifiers()[best])])
        pipe.fit(X_tr, y_tr)
        val_f1 = f1_score(y_va, pipe.predict(X_va), average="weighted")
        print(f"  --> Best single: {best}, Val F1w = {val_f1:.3f}")

        for _, r in res.iterrows():
            all_rows.append((bname, r["clf"], r["f1_mean"], r["f1_std"],
                             val_f1 if r["clf"] == best else None))
        per_bb[bname] = (best, X_tr, y_tr, X_va, y_va, val_f1)

    summary = pd.DataFrame(all_rows, columns=["backbone","clf","cv_f1","cv_f1_std","val_f1"])
    summary = summary.sort_values("cv_f1", ascending=False).reset_index(drop=True)
    print(f"\n{'='*60}\nRANKING (by CV F1w)\n{'='*60}")
    print(summary.to_string(index=False))

    # Pick backbone whose best-single *val* F1 is highest (honest held-out metric)
    best_bb = max(per_bb, key=lambda b: per_bb[b][5])
    best_clf_name, X_tr, y_tr, X_va, y_va, single_f1 = per_bb[best_bb]
    print(f"\nBest backbone by Val F1w: {best_bb} ({best_clf_name} = {single_f1:.3f})")

    # ---------- Stacking on winning backbone ----------
    print(f"\n{'='*60}\nSTACKING on {best_bb}\n{'='*60}")
    ests = [(n, Pipeline([("scaler", StandardScaler()), ("clf", c)]))
            for n, c in get_classifiers().items()]
    stack = StackingClassifier(
        estimators=ests,
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        stack_method="predict_proba", n_jobs=-1,
    )
    stack.fit(X_tr, y_tr)
    stack_f1 = f1_score(y_va, stack.predict(X_va), average="weighted")
    print(f"Stacking Val F1w = {stack_f1:.3f}")

    if stack_f1 >= single_f1:
        final_model, final_name, final_f1 = stack, "stacking", stack_f1
    else:
        single_pipe = Pipeline([("scaler", StandardScaler()),
                                ("clf", get_classifiers()[best_clf_name])])
        single_pipe.fit(X_tr, y_tr)
        final_model, final_name, final_f1 = single_pipe, best_clf_name, single_f1

    print(f"\nFINAL: {best_bb} + {final_name}, Val F1w = {final_f1:.3f}")
    print(classification_report(y_va, final_model.predict(X_va), target_names=le.classes_))
    plot_confusion(y_va, final_model.predict(X_va), le.classes_,
                   f"{best_bb} + {final_name} (Val F1w={final_f1:.3f})",
                   f"confmat_{best_bb}_{final_name}.png")

    # ---------- Save ----------
    joblib.dump({"model": final_model, "label_encoder": le}, "model.joblib")
    with open("model_meta.json", "w") as f:
        json.dump({
            "backbone_name": best_bb,
            "backbone_timm_id": BACKBONES[best_bb],
            "classifier": final_name,
            "final_size": 256,
            "classes": le.classes_.tolist(),
            "val_f1_weighted": float(final_f1),
            "feature_l2_normalize": True,
            "scale_is_feature": True,
            "n_tta": N_TTA,
        }, f, indent=2)
    summary.to_csv("benchmark_results_v2.csv", index=False)
    print("\nSaved: model.joblib, model_meta.json, benchmark_results_v2.csv")