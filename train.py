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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ---------- 1. Load data ----------
def load_data():
    # EDIT THIS to load your dataframe
    df = pd.read_pickle("tears.pkl")
    return df


# ---------- 2. Feature extraction with pretrained ResNet50 ----------
def extract_features(df, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()  # output 2048-d embedding
    model.eval().to(device)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    feats = []
    with torch.no_grad():
        for img in df["image"]:
            x = tfm(img.astype(np.uint8)).unsqueeze(0).to(device)
            feats.append(model(x).cpu().numpy().squeeze())
    X = np.vstack(feats)
    # append scale as an extra feature
    X = np.hstack([X, df["scale"].values.reshape(-1, 1)])
    return X


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
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
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
    y = LabelEncoder().fit_transform(df["label"])
    print("Extracting ResNet50 features...")
    X = extract_features(df)
    print(f"Feature matrix: {X.shape}")
    print("\nBenchmarking classifiers (5-fold stratified CV, weighted F1):\n")
    results = benchmark(X, y)
    results.to_csv("benchmark_results.csv", index=False)