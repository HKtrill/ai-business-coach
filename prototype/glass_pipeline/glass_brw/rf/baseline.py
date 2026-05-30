"""
glass_brw.rf.baseline
=====================
Baseline Random Forest for comparison against the Optuna-tuned RF stage.

Mirrors Cell 11B — default params, no tuning, same binary feature input.
Used to compute baseline_auc passed into train_rf_stage().

Public API
----------
train_rf_baseline(X_train, X_test, y_train, y_test) → dict
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def train_rf_baseline(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    n_estimators:     int = 200,
    max_depth:        int = 10,
    min_samples_leaf: int = 50,
    random_state:     int = 42,
) -> dict:
    """
    Train a default-param RF on binary features and return test metrics.

    Parameters
    ----------
    X_train, X_test   : 29-column binary DataFrames from engineer_features()
    y_train, y_test   : binary target Series
    n_estimators      : number of trees (default 200)
    max_depth         : max tree depth (default 10)
    min_samples_leaf  : min samples per leaf (default 50)
    random_state      : global seed

    Returns
    -------
    metrics : dict
        Keys: accuracy, roc_auc, precision, recall, f1, brier
    """
    print("\n" + "=" * 80)
    print("🌲 RF STAGE 2 — BASELINE (default params, no tuning)")
    print("=" * 80)
    print(f"   n_estimators={n_estimators} | max_depth={max_depth} | "
          f"min_samples_leaf={min_samples_leaf} | class_weight=balanced")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    proba = rf.predict_proba(X_test)[:, 1]
    pred  = rf.predict(X_test)

    metrics = {
        "accuracy":  float(accuracy_score(y_test, pred)),
        "roc_auc":   float(roc_auc_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall":    float(recall_score(y_test, pred)),
        "f1":        float(f1_score(y_test, pred)),
        "brier":     float(brier_score_loss(y_test, proba)),
    }

    print(f"\n   Accuracy  : {metrics['accuracy']:.4f}")
    print(f"   ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"   Precision : {metrics['precision']:.4f}")
    print(f"   Recall    : {metrics['recall']:.4f}")
    print(f"   F1        : {metrics['f1']:.4f}")
    print(f"   Brier     : {metrics['brier']:.4f}")
    print("=" * 80)

    return metrics