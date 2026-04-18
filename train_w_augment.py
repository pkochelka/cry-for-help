"""
Benchmark multiple pretrained feature extractors + classifiers on tear microscopy images
(300 imbalanced samples, 5 classes).

Strategy:
  - Too few samples to train a CNN from scratch -> use frozen pretrained backbones
    as feature extractors, then train classical classifiers on the embeddings + scale.
  - Compare multiple backbones (ResNet50, ConvNeXt-V2, EfficientNet-V2, Swin-V2, DINOv2).
  - Stratified CV, class_weight='balanced' to handle imbalance.
  - Random oversampling of minority classes, each re-passed through augmentations.
  - Scored by weighted F1 + confusion matrix.

Install: pip install timm
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from augmentation import build_eval_transform, build_train_transform

CACHE_DIR = "features_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_tfm = build_train_transform()
eval_tfm = build_eval_transform()


# ---------- 1. Load data ----------
def load_data():
    df = pd.read_pickle("out.pkl")
    return df


def stratified_split(df, test_size=0.2, random_state=0):
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def random_oversample(df, random_state=0):
    """Upsample minority classes to match the majority class count.
    Duplicates are kept as duplicate rows; augmentation at feature-extraction
    time will produce different embeddings for each copy."""
    rng = np.random.default_rng(random_state)
    counts = df["label"].value_counts()
    max_n = counts.max()
    parts = []
    for label, n in counts.items():
        sub = df[df["label"] == label]
        if n < max_n:
            extra_idx = rng.choice(sub.index.values, size=max_n - n, replace=True)
            sub = pd.concat([sub, df.loc[extra_idx]], axis=0)
        parts.append(sub)
    out = pd.concat(parts, axis=0).sample(frac=1, random_state=random_state)
    return out.reset_index(drop=True)


# ---------- 2. Backbones ----------
def get_backbones():
    return {
        #"resnet50": lambda: _torchvision_resnet50(),
        "convnextv2_base": lambda: _timm_model("convnextv2_base.fcmae_ft_in22k_in1k"),
        #"efficientnetv2_s": lambda: _timm_model("tf_efficientnetv2_s.in21k_ft_in1k"),
        #"swinv2_base": lambda: _timm_model("swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"),
        #"dinov2_base": lambda: _timm_model("vit_base_patch14_dinov2.lvd142m"),
    }


def _torchvision_resnet50():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Identity()
    return m.eval().to(DEVICE)


def _timm_model(name):
    m = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")
    return m.eval().to(DEVICE)


# ---------- 3. Feature extraction ----------
def extract_features(df, model, tfm, n_aug=1):
    feats, labels, scales = [], [], []
    with torch.no_grad():
        for i, row in df.iterrows():
            if i % 25 == 0:
                print(f"  {i}/{len(df)}")
            img = row["pixels"].astype(np.uint8)
            for _ in range(n_aug):
                x = tfm(img).unsqueeze(0).to(DEVICE)
                feat = model(x).cpu().numpy().squeeze()
                feats.append(feat)
                labels.append(row["label"])
                scales.append(row["scale"])
    X = np.vstack(feats)
    X = np.hstack([X, np.array(scales).reshape(-1, 1)])
    return X, np.array(labels)


def load_or_extract(backbone_name, split_name, df, tfm, n_aug, model_builder):
    path = os.path.join(CACHE_DIR, f"{backbone_name}_{split_name}.npz")
    try:
        data = np.load(path)
        print(f"Loaded cached {backbone_name}/{split_name} from {path}")
        return data["X"], data["y"]
    except (FileNotFoundError, OSError):
        print(f"Extracting {backbone_name}/{split_name} features...")
        model = model_builder()
        X, y = extract_features(df, model, tfm=tfm, n_aug=n_aug)
        np.savez(path, X=X, y=y)
        del model
        torch.cuda.empty_cache() if DEVICE == "cuda" else None
        return X, y


# ---------- 4. Classifiers (with pre-tuned hyperparams) ----------
def get_classifiers():
    return {
        "LogReg (L2)": LogisticRegression(
            C=10.0, max_iter=2000, class_weight="balanced",
            solver="lbfgs", random_state=0,
        ),
        "LogReg (L1)": LogisticRegression(
            C=0.5, max_iter=2000, class_weight="balanced", l1_ratio=0.0, solver="saga", random_state=1,
        ),
        "SVM (linear)": SVC(
            C=1.0, kernel="linear", class_weight="balanced",
            probability=True, random_state=2,
        ),
    }


# ---------- 5. Benchmark one backbone ----------
def benchmark(X, y):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    results = []

    for name, clf in get_classifiers().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        results.append((name, scores.mean(), scores.std()))
        print(f"    {name:15s}  F1w = {scores.mean():.3f} ± {scores.std():.3f}")

    results.sort(key=lambda r: -r[1])
    return pd.DataFrame(results, columns=["clf", "f1_mean", "f1_std"])


def plot_confusion(y_true, y_pred, class_names, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d",
    )
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved confusion matrix -> {out_path}")
    print("  Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())


# ---------- 6. Main ----------
if __name__ == "__main__":
    df = load_data()
    train_df, val_df = stratified_split(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Oversample minority classes (duplicates will get different augmentations)
    #print(f"Class counts before oversampling: {train_df['label'].value_counts().to_dict()}")
    #train_df = random_oversample(train_df, random_state=0)
    #print(f"Class counts after  oversampling: {train_df['label'].value_counts().to_dict()}\n")

    le = LabelEncoder()
    le.fit(train_df["label"])

    backbones = get_backbones()
    all_results = []
    per_backbone_best = {}

    for bname, builder in backbones.items():
        print(f"\n{'='*60}\nBackbone: {bname}\n{'='*60}")
        try:
            X_tr, y_tr_raw = load_or_extract(bname, "train", train_df, train_tfm, 3, builder)
            X_va, y_va_raw = load_or_extract(bname, "val",   val_df,   eval_tfm,  1, builder)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")
            continue

        y_tr = le.transform(y_tr_raw)
        y_va = le.transform(y_va_raw)
        print(f"  Train: {X_tr.shape}, Val: {X_va.shape}\n")

        res_df = benchmark(X_tr, y_tr)

        # Eval best classifier for this backbone on val set
        best_clf_name = res_df.iloc[0]["clf"]
        clf = get_classifiers()[best_clf_name]
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        y_va_pred = pipe.predict(X_va)
        val_f1 = f1_score(y_va, y_va_pred, average="weighted")
        print(f"  --> Best clf: {best_clf_name}, Val F1w = {val_f1:.3f}")

        plot_confusion(
            y_va, y_va_pred, le.classes_,
            title=f"{bname} + {best_clf_name} (Val F1w={val_f1:.3f})",
            out_path=f"confmat_{bname}_{best_clf_name.replace(' ', '_')}.png",
        )

        for _, row in res_df.iterrows():
            all_results.append((bname, row["clf"], row["f1_mean"], row["f1_std"],
                                val_f1 if row["clf"] == best_clf_name else None))

        per_backbone_best[bname] = (best_clf_name, X_tr, y_tr, X_va, y_va)

    # ---------- Summary ----------
    summary = pd.DataFrame(all_results, columns=["backbone", "clf", "cv_f1_mean", "cv_f1_std", "val_f1"])
    summary = summary.sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)
    print(f"\n{'='*60}\nFULL RANKING (by CV F1w)\n{'='*60}")
    print(summary.to_string(index=False))

    best = summary.iloc[0]
    print(f"\nWinner: {best['backbone']} + {best['clf']} (CV F1w = {best['cv_f1_mean']:.3f})")

    # ---------- Stacking on best backbone ----------
    print(f"\n{'='*60}\nSTACKING on best backbone\n{'='*60}")
    Xw_tr = per_backbone_best[best["backbone"]][1]
    yw_tr = per_backbone_best[best["backbone"]][2]
    Xw_va = per_backbone_best[best["backbone"]][3]
    yw_va = per_backbone_best[best["backbone"]][4]

    est2 = []
    for cname, clf in get_classifiers().items():
        p = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        est2.append((cname.replace(" ", "_"), p))
    stacking2 = StackingClassifier(
        estimators=est2,
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        stack_method="predict_proba",
        n_jobs=-1,
    )
    stacking2.fit(Xw_tr, yw_tr)
    yw_va_pred = stacking2.predict(Xw_va)
    stack_f1 = f1_score(yw_va, yw_va_pred, average="weighted")
    print(f"Stacking on {best['backbone']} features: Val F1w = {stack_f1:.3f}")

    # ---------- Final report ----------
    print(f"\n{'='*60}\nFINAL CLASSIFICATION REPORT ({best['backbone']} + stacking)\n{'='*60}")
    print(classification_report(yw_va, yw_va_pred, target_names=le.classes_))

    plot_confusion(
        yw_va, yw_va_pred, le.classes_,
        title=f"{best['backbone']} + stacking (Val F1w={stack_f1:.3f})",
        out_path=f"confmat_{best['backbone']}_stacking.png",
    )

    summary.to_csv("benchmark_results.csv", index=False)
    print("\nSaved results to benchmark_results.csv")