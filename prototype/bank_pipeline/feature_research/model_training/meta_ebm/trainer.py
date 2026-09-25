"""
model_training/meta_ebm/trainer.py
------------------------------------
Balanced Optuna parameter constants and model training helpers
for the meta-arbiter ensemble (LR + RF + EBM).

Parameter source
----------------
LR  — Balanced Optuna (100 trials, 5-fold CV)   AUC: 0.7822
RF  — Balanced Optuna (200 trials, 5-fold CV)   AUC: 0.7933
EBM — ⚠️  PLACEHOLDER — replace when balanced tuning completes
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, roc_auc_score,
)
from .arbiter import find_recall_threshold


# ══════════════════════════════════════════════════════════════════════════════
# BALANCED OPTUNA PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# LR — Balanced (Cell 12B balanced)
# C=0.000285 gives strong regularisation; l2 + lbfgs is stable at this scale
LR_PARAMS: dict = dict(
    C=0.000285,
    penalty="l2",
    class_weight="balanced",
    solver="lbfgs",
    max_iter=5000,
    random_state=42,
)

# RF — Balanced binary (Cell 13B balanced, 29-bin features)
# minority_weight=1.0064 ≈ 1.0 so class_weight='balanced' is the effective weight
RF_PARAMS: dict = dict(
    n_estimators=650,
    max_depth=7,
    min_samples_leaf=26,
    max_features=0.6519,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# EBM — Recall-Biased Optuna (100 trials, 5-fold CV)  
# AUC: 0.8018  F2: 0.5604
EBM_PARAMS: dict = dict(
    learning_rate=0.022187,
    max_rounds=4900,
    max_bins=480,
    max_interaction_bins=32,
    interactions=2,
    n_jobs=1,
    random_state=42,
)

# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL MODEL TRAINERS
# ══════════════════════════════════════════════════════════════════════════════

def train_lr(
    df,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    features: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    params: dict | None = None,
    target_recall: float = 0.70,
    verbose: bool = True,
) -> tuple:
    """
    Train Logistic Regression on continuous features with StandardScaler.

    Parameters
    ----------
    df          : df_engineered (continuous features)
    train_idx   : integer indices into df for the training split
    test_idx    : integer indices into df for the test split
    features    : LR_FEATURES list
    y_train/y_test : label arrays
    params      : override LR_PARAMS if provided
    target_recall : recall target for threshold search on train set
    verbose     : print metrics + coefficients

    Returns
    -------
    model, scaler, p_train, p_test, threshold, pred_test
    """
    params = params or LR_PARAMS
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df.iloc[train_idx][features].values)
    X_te = scaler.transform(df.iloc[test_idx][features].values)

    model = LogisticRegression(**params)
    model.fit(X_tr, y_train)

    p_tr  = model.predict_proba(X_tr)[:, 1]
    p_te  = model.predict_proba(X_te)[:, 1]
    thresh = find_recall_threshold(y_train, p_tr, target_recall)
    pred_te = (p_te >= thresh).astype(int)

    if verbose:
        print(
            f"   🔵 LR  thresh={thresh:.4f}  "
            f"Acc={accuracy_score(y_test, pred_te):.4f}  "
            f"Rec={recall_score(y_test, pred_te):.4f}  "
            f"Prec={precision_score(y_test, pred_te):.4f}  "
            f"AUC={roc_auc_score(y_test, p_te):.4f}"
        )
        for feat, coef in sorted(
            zip(features, model.coef_[0]), key=lambda x: -abs(x[1])
        ):
            print(f"      {'➕' if coef > 0 else '➖'} {feat:<35} {coef:>+.4f}")

    return model, scaler, p_tr, p_te, thresh, pred_te


def train_rf(
    df,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    features: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    params: dict | None = None,
    target_recall: float = 0.70,
    verbose: bool = True,
) -> tuple:
    """
    Train Random Forest on binary features (df_binned).

    Parameters
    ----------
    df          : df_binned (29 binary features)
    features    : RF_FEATURES_BINARY list

    Returns
    -------
    model, p_train, p_test, threshold, pred_test
    """
    params = params or RF_PARAMS
    X_tr = df.iloc[train_idx][features].values
    X_te = df.iloc[test_idx][features].values

    model = RandomForestClassifier(**params)
    model.fit(X_tr, y_train)

    p_tr  = model.predict_proba(X_tr)[:, 1]
    p_te  = model.predict_proba(X_te)[:, 1]
    thresh = find_recall_threshold(y_train, p_tr, target_recall)
    pred_te = (p_te >= thresh).astype(int)

    if verbose:
        print(
            f"   🌲 RF  thresh={thresh:.4f}  "
            f"Acc={accuracy_score(y_test, pred_te):.4f}  "
            f"Rec={recall_score(y_test, pred_te):.4f}  "
            f"Prec={precision_score(y_test, pred_te):.4f}  "
            f"AUC={roc_auc_score(y_test, p_te):.4f}"
        )

    return model, p_tr, p_te, thresh, pred_te


def train_ebm(
    df,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    features: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    params: dict | None = None,
    target_recall: float = 0.70,
    verbose: bool = True,
) -> tuple:
    """
    Train Explainable Boosting Machine on engineered features.

    Parameters
    ----------
    df          : df_engineered (EBM feature set)
    features    : EBM_FEATURES list
    params      : override EBM_PARAMS if provided

    Returns
    -------
    model, p_train, p_test, threshold, pred_test, success (bool)

    Notes
    -----
    Returns (None, None, None, 0.5, None, False) if interpret not installed
    or if EBM_PARAMS still contains None placeholders.
    """
    params = params or EBM_PARAMS

    # Guard: block training if placeholder params are still None
    if any(v is None for v in params.values()):
        missing = [k for k, v in params.items() if v is None]
        print(f"   ⚠️  EBM skipped — placeholder params not yet filled: {missing}")
        return None, None, None, 0.50, None, False

    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except ImportError:
        print("   ⚠️  interpret not installed — skipping EBM")
        return None, None, None, 0.50, None, False

    X_tr = df.iloc[train_idx][features].values
    X_te = df.iloc[test_idx][features].values

    model = ExplainableBoostingClassifier(**params)
    model.fit(X_tr, y_train)

    p_tr  = model.predict_proba(X_tr)[:, 1]
    p_te  = model.predict_proba(X_te)[:, 1]
    thresh = find_recall_threshold(y_train, p_tr, target_recall)
    pred_te = (p_te >= thresh).astype(int)

    if verbose:
        print(
            f"   🔮 EBM thresh={thresh:.4f}  "
            f"Acc={accuracy_score(y_test, pred_te):.4f}  "
            f"Rec={recall_score(y_test, pred_te):.4f}  "
            f"Prec={precision_score(y_test, pred_te):.4f}  "
            f"AUC={roc_auc_score(y_test, p_te):.4f}"
        )

    return model, p_tr, p_te, thresh, pred_te, True


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(
    df_continuous,
    df_binary,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    lr_features: list[str],
    rf_features: list[str],
    ebm_features: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_recall: float = 0.70,
    verbose: bool = True,
) -> dict:
    """
    Train LR + RF + EBM and return a unified results dict.

    Parameters
    ----------
    df_continuous : df_engineered — used by LR and EBM
    df_binary     : df_binned    — used by RF (29 binary features)

    Returns
    -------
    dict with keys 'lr', 'rf', 'ebm', each containing:
        model, p_train, p_test, thresh, pred, (scaler for LR), ok (EBM only)
    """
    if verbose:
        print("\n" + "-" * 80)
        print("TRAINING MODELS — balanced Optuna params")
        print("-" * 80)

    lr_model, lr_scaler, lr_ptr, lr_pte, lr_thresh, lr_pred = train_lr(
        df_continuous, train_idx, test_idx, lr_features,
        y_train, y_test, target_recall=target_recall, verbose=verbose,
    )

    rf_model, rf_ptr, rf_pte, rf_thresh, rf_pred = train_rf(
        df_binary, train_idx, test_idx, rf_features,
        y_train, y_test, target_recall=target_recall, verbose=verbose,
    )

    ebm_model, ebm_ptr, ebm_pte, ebm_thresh, ebm_pred, has_ebm = train_ebm(
        df_continuous, train_idx, test_idx, ebm_features,
        y_train, y_test, target_recall=target_recall, verbose=verbose,
    )

    return {
        "lr": {
            "model":   lr_model,
            "scaler":  lr_scaler,
            "p_train": lr_ptr,
            "p_test":  lr_pte,
            "thresh":  lr_thresh,
            "pred":    lr_pred,
        },
        "rf": {
            "model":   rf_model,
            "p_train": rf_ptr,
            "p_test":  rf_pte,
            "thresh":  rf_thresh,
            "pred":    rf_pred,
        },
        "ebm": {
            "model":   ebm_model,
            "p_train": ebm_ptr,
            "p_test":  ebm_pte,
            "thresh":  ebm_thresh,
            "pred":    ebm_pred,
            "ok":      has_ebm,
        },
    }