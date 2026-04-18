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
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from augmentation import build_eval_transform, build_train_transform

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


# ---------- 3. Classifiers to compare ----------
def get_classifiers():
    return {
        "LogReg (L2)":      LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "LogReg (L1)":      LogisticRegression(max_iter=2000, class_weight="balanced",
                                               penalty="l1", solver="saga", C=0.5, random_state=0),
        "SVM (RBF)":        SVC(kernel="rbf", class_weight="balanced", C=1.0, gamma="scale"),
        "SVM (linear)":     SVC(kernel="linear", class_weight="balanced", C=1.0, random_state=0),
        "kNN":              KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Random Forest":    RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                                   random_state=0),
        "Gradient Boost":   GradientBoostingClassifier(n_estimators=300, random_state=0),
        "XGBoost":          XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                          eval_metric="mlogloss", random_state=0),
        "LightGBM":         LGBMClassifier(n_estimators=400, max_depth=-1, learning_rate=0.05,
                                           class_weight="balanced", random_state=0, verbose=-1),
    }


# ---------- 4. Benchmark ----------
def benchmark(X, y):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    results = []
    for name, clf in get_classifiers().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        results.append((name, scores.mean(), scores.std()))
        print(f"{name:18s}  F1w = {scores.mean():.3f} ± {scores.std():.3f}")
    results.sort(key=lambda r: -r[1])
    print("\nBest:", results[0][0])
    return pd.DataFrame(results, columns=["model", "f1_weighted_mean", "f1_weighted_std"])


if __name__ == "__main__":
    df = load_data()

    # --- split first ---
    train_df, val_df = stratified_split(df)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # --- encode labels ---
    le = LabelEncoder()
    y_train_raw = le.fit_transform(train_df["label"])
    y_val_raw = le.transform(val_df["label"])

    # --- extract features ---
    print("Extracting TRAIN features with augmentation...")
    X_train, y_train = extract_features(
        train_df,
        tfm=train_tfm,
        n_aug=5  # 🔥 important: increases dataset size
    )

    print("Extracting VAL features (no augmentation)...")
    X_val, y_val = extract_features(
        val_df,
        tfm=eval_tfm,
        n_aug=1
    )

    print("Train features:", X_train.shape)
    print("Val features:", X_val.shape)

    # --- benchmark on TRAIN using CV ---
    print("\nBenchmarking classifiers on TRAIN (CV):\n")
    results = benchmark(X_train, y_train)

    # --- final evaluation on held-out VAL ---
    print("\nFinal evaluation on validation set:\n")

    best_model_name = results.iloc[0]["model"]
    clf = get_classifiers()[best_model_name]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)

    from sklearn.metrics import f1_score
    f1 = f1_score(y_val, y_pred, average="weighted")

    print(f"Best model: {best_model_name}")
    print(f"Validation F1 (weighted): {f1:.3f}")