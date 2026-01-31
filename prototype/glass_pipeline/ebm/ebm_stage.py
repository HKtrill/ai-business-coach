from pyexpat import model
import numpy as np
import pandas as pd

from .feature_engineering import (
    drop_leaky_features,
    engineer_ebm_features,
    prune_redundant_features,
)
from .interactions import define_ebm_interactions
from .tuning import tune_ebm
from .evaluation import evaluate_ebm
from .calibration import calibrate_if_needed
from .artifacts import save_ebm_artifacts


def train_ebm_stage(GLOBAL_SPLIT):
    """
    Train Stage 3 Explainable Boosting Machine (EBM).

    Saves a SINGLE canonical artifact containing:
    - raw + calibrated probabilities (train/test)
    - model + calibrated model
    - threshold
    - metadata required for Meta-EBM routing
    """

    # =========================
    # DATA
    # =========================
    X_train = GLOBAL_SPLIT['X_train'].copy()
    X_test  = GLOBAL_SPLIT['X_test'].copy()
    y_train = GLOBAL_SPLIT['y_train']
    y_test  = GLOBAL_SPLIT['y_test']

    # =========================
    # FEATURE ENGINEERING
    # =========================
    X_train, X_test = drop_leaky_features(X_train, X_test)
    X_train, X_test, _ = engineer_ebm_features(X_train, X_test)
    X_train, X_test = prune_redundant_features(X_train, X_test)

    # =========================
    # INTERACTIONS
    # =========================
    interactions = define_ebm_interactions(X_train)

    # =========================
    # TRAIN EBM
    # =========================
    model, best_params, best_cv = tune_ebm(
        X_train,
        y_train,
        interactions
    )

    # =========================
    # EVALUATION
    # =========================
    metrics = evaluate_ebm(model, X_test, y_test)

    # =========================
    # CALIBRATION
    # =========================
    calibrated_model, train_probs_cal, test_probs_cal, ece = calibrate_if_needed(
        model,
        X_train, y_train,
        X_test, y_test
    )

    # =========================
    # PROBABILITIES (REQUIRED)
    # =========================
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs  = model.predict_proba(X_test)[:, 1]

    if calibrated_model is not None:
        train_probs_cal = calibrated_model.predict_proba(X_train)[:, 1]
        test_probs_cal  = calibrated_model.predict_proba(X_test)[:, 1]
    else:
        train_probs_cal = train_probs
        test_probs_cal  = test_probs

    # =========================
    # ARTIFACT (CANONICAL)
    # =========================
    artifact = {
        'model': model,
        'calibrated_model': calibrated_model,

        'train_predictions': train_probs,
        'test_predictions': test_probs,
        'train_predictions_calibrated': train_probs_cal,
        'test_predictions_calibrated': test_probs_cal,

        'optimal_threshold': 0.5,

        'features': X_train.columns.tolist(),
        'interactions': interactions,
        'best_params': best_params,
        'cv_score': best_cv,
        'metrics': metrics,
        'ece': ece,
    }


    # =========================
    # SAVE
    # =========================
    path = save_ebm_artifacts(artifact)

    return artifact, path

