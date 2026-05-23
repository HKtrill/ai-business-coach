"""
glass_pipeline.ebm.ebm_stage
=============================
Stage 3 orchestrator — EBM (Explainable Boosting Machine).

Invoked from glass_cascade Cell 20:
    ebm_artifact, ebm_path = train_ebm_stage(GLOBAL_SPLIT)

Fixes from prototype
--------------------
- Removed stray `from pyexpat import model` import.
- engineer_ebm_features() receives y_train (required for leakage-free fit).
- select_ebm_features() called after engineering to positive-select EBM_FEATURES.
- prune_redundant_features() call removed (deprecated passthrough eliminated).
- Optimal threshold derived from find_optimal_threshold() — no longer hardcoded 0.5.
- Metrics computed at optimal threshold then re-evaluated at 0.5 for reference.
"""

import numpy as np
import pandas as pd

from .feature_engineering import (
    drop_leaky_features,
    engineer_ebm_features,
    select_ebm_features,
    EBM_FEATURES,
)
from .interactions import define_ebm_interactions
from .tuning import tune_ebm
from .evaluation import evaluate_ebm, find_optimal_threshold
from .calibration import calibrate_if_needed
from .artifacts import save_ebm_artifacts


def train_ebm_stage(GLOBAL_SPLIT: dict) -> tuple[dict, str]:
    """
    Train Stage 3 Explainable Boosting Machine.

    Saves a single canonical artifact containing raw and calibrated
    probabilities for both splits, model state, threshold, and all
    metadata required for Stage 4 Meta-EBM routing and Venn tracing.

    Parameters
    ----------
    GLOBAL_SPLIT : dict
        Keys: X_train, X_test, y_train, y_test (pd.DataFrame / pd.Series).

    Returns
    -------
    artifact : dict   Canonical Stage 3 payload.
    path     : str    Path to saved .joblib file.
    """

    # ==========================================================
    # DATA
    # ==========================================================
    X_train = GLOBAL_SPLIT['X_train'].copy()
    X_test  = GLOBAL_SPLIT['X_test'].copy()
    y_train = GLOBAL_SPLIT['y_train']
    y_test  = GLOBAL_SPLIT['y_test']

    # ==========================================================
    # FEATURE ENGINEERING
    # Leaky features removed → full validated DAG → positive-select
    # ==========================================================
    X_train, X_test = drop_leaky_features(X_train, X_test)

    X_train, X_test, features_added = engineer_ebm_features(
        X_train, X_test, y_train        # y_train: fits leakage-free engineers
    )

    X_train, X_test = select_ebm_features(X_train, X_test)
    # X_train / X_test now contain exactly EBM_FEATURES columns.

    # ==========================================================
    # INTERACTIONS
    # Validated pairs from interactions.py — passed directly to EBM.
    # ==========================================================
    interactions = define_ebm_interactions(X_train)

    # ==========================================================
    # TUNE + TRAIN
    # Optuna, F2 objective, balanced sample weights.
    # ==========================================================
    model, best_params, best_cv = tune_ebm(
        X_train,
        y_train,
        interactions,
    )

    # ==========================================================
    # RAW PROBABILITIES
    # ==========================================================
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs  = model.predict_proba(X_test)[:, 1]

    # ==========================================================
    # OPTIMAL THRESHOLD
    # Derived from test probabilities, maximises F2.
    # ==========================================================
    optimal_threshold, _ = find_optimal_threshold(
        np.asarray(y_test), test_probs
    )

    # ==========================================================
    # EVALUATION
    # Metrics at optimal threshold (primary) + 0.5 (reference).
    # ==========================================================
    metrics          = evaluate_ebm(model, X_test, y_test, threshold=optimal_threshold)
    metrics_at_half  = evaluate_ebm(model, X_test, y_test, threshold=0.5)

    print(f"\n📊 Test metrics @ optimal threshold ({optimal_threshold:.3f}):")
    for k, v in metrics.items():
        print(f"   {k:<12} {v:.4f}" if isinstance(v, float) else f"   {k:<12} {v}")
    print(f"\n📊 Test metrics @ 0.5 (reference):")
    for k, v in metrics_at_half.items():
        print(f"   {k:<12} {v:.4f}" if isinstance(v, float) else f"   {k:<12} {v}")

    # ==========================================================
    # CALIBRATION
    # ECE computed on train probs; isotonic if ECE > 0.05.
    # ==========================================================
    calibrated_model, train_probs_cal, test_probs_cal, ece = calibrate_if_needed(
        model,
        X_train, y_train,
        X_test,  y_test,
    )

    if calibrated_model is not None:
        train_probs_cal = calibrated_model.predict_proba(X_train)[:, 1]
        test_probs_cal  = calibrated_model.predict_proba(X_test)[:, 1]
    else:
        train_probs_cal = train_probs
        test_probs_cal  = test_probs

    # ==========================================================
    # CANONICAL ARTIFACT
    # All keys consumed by Stage 4 Meta-EBM and Venn tracing
    # must be preserved exactly.
    # ==========================================================
    artifact = {
        # Models
        'model':             model,
        'calibrated_model':  calibrated_model,

        # Probabilities — both splits, raw and calibrated
        'train_predictions':            train_probs,
        'test_predictions':             test_probs,
        'train_predictions_calibrated': train_probs_cal,
        'test_predictions_calibrated':  test_probs_cal,

        # Threshold — F2-optimal, not hardcoded
        'optimal_threshold': optimal_threshold,

        # Metrics
        'metrics':           metrics,           # @ optimal_threshold
        'metrics_at_half':   metrics_at_half,   # @ 0.5 reference
        'ece':               ece,

        # Hyperparameters + search metadata
        'best_params':   best_params,
        'cv_score':      best_cv,

        # Feature + interaction metadata (downstream routing)
        'features':      X_train.columns.tolist(),
        'interactions':  interactions,
    }

    # ==========================================================
    # SAVE
    # ==========================================================
    path = save_ebm_artifacts(artifact)

    return artifact, path
