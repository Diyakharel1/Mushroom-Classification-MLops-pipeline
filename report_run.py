#!/usr/bin/env python3
"""
Train once for the report:
- Loads processed parquet (fallback: raw CSV + ';' delimiter)
- Trains XGBoost (fallback: RandomForest if xgboost not installed)
- Prints metrics to console
- Saves confusion matrix + ROC curve to artifacts/
"""

import os
from pathlib import Path
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier

# Try XGBoost
try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False

ART = Path("artifacts"); ART.mkdir(parents=True, exist_ok=True)
PROCESSED = Path("data/processed/secondary_transformed.parquet")
RAW = Path("data/raw/secondary_data.csv")

def load_data():
    if PROCESSED.exists():
        df = pd.read_parquet(PROCESSED)
        target = "class_encoded" if "class_encoded" in df.columns else "class"
        X = df.drop(columns=[target])
        y = df[target]
        if y.dtype == object:
            y = (y.astype(str).str.lower().isin(["p","poison","poisonous"])).astype(int)
        return X, y, "processed_parquet"
    else:
        if not RAW.exists():
            raise FileNotFoundError(f"Missing data: {PROCESSED} and {RAW}")
        df_raw = pd.read_csv(RAW, sep=";")
        # Minimal inline transform for target only (you already have a full transform pipeline elsewhere)
        y = (df_raw["class"].astype(str).str.lower().isin(["p","poison","poisonous"])).astype(int)
        # Use a tiny numeric subset so this fallback still runs
        X = df_raw[["cap_diameter","stem_height","stem_width"]].copy()
        return X, y, "raw_csv_minimal"

def build_model():
    if XGB_OK:
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, tree_method="hist"
        )
        name = "xgboost"
    else:
        model = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42)
        name = "random_forest"
    return name, model

def main():
    X, y, source = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_name, model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "data_source": source,
        "model": model_name,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "class_balance_test": {int(k): float(v) for k,v in y_test.value_counts(normalize=True).to_dict().items()},
    }

    # Save metrics JSON
    (ART / "report_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # Confusion matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(ART / "confusion_matrix.png"); plt.close()

    # ROC curve (if proba available)
    try:
        proba = getattr(model, "predict_proba", None)
        if proba is not None:
            p = proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, p)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
            plt.plot([0,1],[0,1],"--")
            plt.title("ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
            plt.tight_layout(); plt.savefig(ART / "roc_curve.png"); plt.close()
    except Exception:
        pass

    print("✅ Saved: artifacts/confusion_matrix.png", "(and artifacts/roc_curve.png if supported)")

if __name__ == "__main__":
    main()
