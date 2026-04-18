"""
Benchmark multiple classifiers on tear microscopy images (300 imbalanced samples, 5 classes).

Strategy:
  - Too few samples to train a CNN from scratch -> use a frozen pretrained CNN
    (ResNet50, ImageNet) as a feature extractor, then train classical classifiers
    on the 2048-d embeddings + scale feature.
  - Stratified 5-fold CV, class_weight='balanced' to handle imbalance.
  - Scored by weighted F1 (the user's target metric).

Expected input: a pandas DataFrame `df` with columns ['image', 'label', 'scale'],
loadable however you normally load it. Edit `load_data()` below.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, VotingClassifier
from augmentation import build_eval_transform, build_train_transform
import os
from sklearn.metrics import f1_score, classification_report

CACHE_DIR = "features_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

train_tfm = build_train_transform()
eval_tfm = build_eval_transform()


# ---------- 1. Load data ----------
def load_data():
    # EDIT THIS to load your dataframe
    df = pd.read_pickle("out.pkl")
    return df


def stratified_split(df, test_size=0.2, random_state=0):
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

# ---------- 2. Feature extraction with pretrained ResNet50 ----------
def extract_features(df, tfm, n_aug=1, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.eval().to(device)

    feats = []
    labels = []
    scales = []

    with torch.no_grad():
        for i, row in df.iterrows():
            print(i/len(df)*100)
            img = row["pixels"].astype(np.uint8)

            for _ in range(n_aug):
                x = tfm(img).unsqueeze(0).to(device)
                feat = model(x).cpu().numpy().squeeze()

                feats.append(feat)
                labels.append(row["label"])
                scales.append(row["scale"])

    X = np.vstack(feats)
    X = np.hstack([X, np.array(scales).reshape(-1, 1)])
    y = np.array(labels)

    return X, y


def load_or_extract(name, df, tfm, n_aug):
    path = os.path.join(CACHE_DIR, f"{name}.npz")
    try:
        data = np.load(path)
        print(f"Loaded cached {name} features from {path}")
        return data["X"], data["y"]
    except (FileNotFoundError, OSError):
        print(f"Extracting {name} features...")
        X, y = extract_features(df, tfm=tfm, n_aug=n_aug)
        np.savez(path, X=X, y=y)
        return X, y



# ---------- 3. Classifiers to compare ----------
def get_classifiers():
    return {
        "LogReg (L2)": (
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=0),
            {"clf__C": [10.0]},
        ),
        "LogReg (L1)": (
            LogisticRegression(max_iter=2000, class_weight="balanced", l1_ratio=0.0, solver="saga", random_state=0),
            {"clf__C": [0.5]},
        ),
        "SVM (linear)": (
            SVC(kernel="linear", class_weight="balanced", probability=True, random_state=0),
            {"clf__C": [10.0]},
        ),
    }


# ---------- 4. Benchmark with optional grid search ----------
def benchmark(X, y, inner_cv_splits=3):
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    inner_cv = StratifiedKFold(n_splits=inner_cv_splits, shuffle=True, random_state=1)

    results = []
    best_params_log = {}

    for name, (clf, param_grid) in get_classifiers().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        if param_grid:
            search = GridSearchCV(
                pipe, param_grid, cv=inner_cv,
                scoring="f1_weighted", n_jobs=-1, refit=True,
            )
            # Outer CV with the search object as estimator
            scores = cross_val_score(search, X, y, cv=outer_cv, scoring="f1_weighted", n_jobs=1)
            # Fit once on full data to surface best params
            search.fit(X, y)
            best_params_log[name] = search.best_params_
            best_params_str = str(search.best_params_)
        else:
            scores = cross_val_score(pipe, X, y, cv=outer_cv, scoring="f1_weighted", n_jobs=-1)
            best_params_log[name] = {}
            best_params_str = "defaults"

        results.append((name, scores.mean(), scores.std()))
        print(f"{name:18s}  F1w = {scores.mean():.3f} ± {scores.std():.3f}  params={best_params_str}")

    results.sort(key=lambda r: -r[1])
    print("\nBest:", results[0][0])
    return (
        pd.DataFrame(results, columns=["model", "f1_weighted_mean", "f1_weighted_std"]),
        best_params_log,
    )

# ---------- 5. Main ----------
if __name__ == "__main__":
    df = load_data()
    train_df, val_df = stratified_split(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Encode labels from df (single source of truth)
    le = LabelEncoder()
    le.fit(train_df["label"])

    X_train, y_train_raw = load_or_extract("train", train_df, train_tfm, n_aug=3)
    X_val,   y_val_raw   = load_or_extract("val",   val_df,   eval_tfm,  n_aug=1)

    y_train = le.transform(y_train_raw)
    y_val   = le.transform(y_val_raw)

    print(f"Train features: {X_train.shape}, Val features: {X_val.shape}")

    # Benchmark
    print("\nBenchmarking classifiers on TRAIN (nested CV + grid search):\n")
    results, best_params = benchmark(X_train, y_train)

    # Best params found during benchmark (update these from your run)
    estimators = [
        ("logreg_l2", Pipeline([("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", C=best_params["LogReg (L2)"]["clf__C"]))])),
        ("logreg_l1", Pipeline([("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", penalty="l1", solver="saga", C=best_params["LogReg (L1)"]["clf__C"]))])),
        ("svm_linear", Pipeline([("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", class_weight="balanced", probability=True, C=best_params["SVM (linear)"]["clf__C"]))])),
    ]

    # --- Stacking (LR meta-learner) ---
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )
    stacking.fit(X_train, y_train)
    y_pred_stack = stacking.predict(X_val)
    print(f"Stacking F1w    : {f1_score(y_val, y_pred_stack, average='weighted'):.3f}")

    print("\nStacking classification report:")
    print(classification_report(y_val, y_pred_stack, target_names=le.classes_))

    # Final eval with best model, refitted with its best params
    print("\nFinal evaluation on held-out validation set:\n")
    best_name = results.iloc[0]["model"]
    best_clf, best_grid = get_classifiers()[best_name]

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", best_clf)])

    if best_grid:
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
        pipe = GridSearchCV(pipe, best_grid, cv=inner_cv, scoring="f1_weighted", n_jobs=-1, refit=True)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)

    print(f"Best model     : {best_name}")
    if best_grid:
        print(f"Best params    : {pipe.best_params_}")
    print(f"Validation F1w : {f1_score(y_val, y_pred, average='weighted'):.3f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))