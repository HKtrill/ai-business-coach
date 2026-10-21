"""
shared.stage1.oof_calibration
=============================
Leakage-safe Stage 1 calibration, shared by the GLASS ``CalibratedLRStage``
and the black-box ``CalibratedStage1MLP`` (audit F16 / F19).

Procedure
---------
1. Out-of-fold base scores. On each outer fold (``build_fold_designs``:
   encoder and scaler refit on the fold's training rows) a fresh clone of the
   base model is fitted on the fold's training rows and scores its held-out
   rows. Every training row gets one score from a model that never saw it.
2. Cross-fitted candidates (F16, F19). On each outer fold k, the
   calibration procedure is replayed on fold k's TRAINING rows only: inner
   folds with their own encoders and scalers (``inner_fold_designs``, F19),
   inner base models, inner out-of-fold scores, and a sigmoid and an
   isotonic calibrator fitted on those. Both are applied to fold k's step-1
   scores. Every training row thus gets one calibrated probability per family
   from a base model, preprocessing and calibrator none of which saw its
   label.
3. Per-fold family choice -> the out-of-fold output. Still inside fold k's
   training rows, each family's calibrator is cross-fitted over the inner
   folds and scored by Brier; the winner (ties: sigmoid) supplies fold k's
   held-out probabilities. A fold's own labels therefore cannot reach its
   output even through the family choice — choosing once over all rows (the
   first F16 patch) let them, whenever they tipped a near-tie. The output
   may mix families across folds: it replays the whole procedure, selection
   included, on each fold's training rows.
4. Final calibrator. The family with the lower Brier over all step-2
   probabilities (ties: sigmoid) is fitted on the step-1 scores of all
   training rows and composed with the base model fitted on the full
   training split (the stage's final model). That pair scores the test split,
   whose labels are never used.

Scores are what ``CalibratedClassifierCV`` feeds its calibrators:
``decision_function`` when the base model has one (LR), otherwise
``predict_proba[:, 1]`` (MLP). The calibrators match scikit-learn's: Platt's
method with its prior-smoothed targets for ``sigmoid``, and
``IsotonicRegression(out_of_bounds="clip")`` for ``isotonic``.

Why not CalibratedClassifierCV
------------------------------
Its ensemble members (``ensemble=True``, previously used) are fitted on
internal folds of the matrix it is given, which carried target encodings and
scaling learned on every training row (F19). Fitting each member's encoder on
its own rows would make every member need raw features at test time, while
the stages score the engineered, full-train-encoded test features. scikit-
learn's documented alternative, ``ensemble=False`` (one model fitted on all
rows, one calibrator fitted on cross-validated scores), keeps that contract;
it is implemented here because its cross-validated scores must come from
per-fold preprocessing. ``tests/stage1/test_stage1_calibration.py`` checks
it against ``CalibratedClassifierCV(Pipeline([encoder, scaler, model]),
ensemble=False)``, which refits the preprocessing inside every fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from shared.metrics import calculate_ece

from .calibration_guard import assert_refittable
from .crossfit import FoldDesign, inner_fold_designs

__all__ = [
    "CALIBRATION_FAMILIES",
    "PlattCalibrator",
    "Stage1CalibratedModel",
    "OOFCalibration",
    "make_calibrator",
    "base_scores",
    "calibrate_oof",
]

CALIBRATION_FAMILIES = ("sigmoid", "isotonic")


# ----------------------------------------------------------------------
# calibrators
# ----------------------------------------------------------------------
class PlattCalibrator:
    """
    Platt scaling, as scikit-learn's ``CalibratedClassifierCV(method="sigmoid")``
    fits it: binomial log-loss on prior-smoothed targets, L-BFGS-B, large
    scores rescaled first. ``predict`` returns ``P(y = 1)``.
    """

    def fit(self, scores, y) -> "PlattCalibrator":
        F = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(y).ravel()
        max_abs = float(np.max(np.abs(F)))
        scale = max_abs if max_abs >= 30 else 1.0          # as scikit-learn: rescale large scores
        F = F / scale
        prior0 = float(np.sum(y <= 0))
        prior1 = len(y) - prior0
        T = np.where(y > 0, (prior1 + 1.0) / (prior1 + 2.0), 1.0 / (prior0 + 2.0))

        def loss_grad(AB):
            raw = -(AB[0] * F + AB[1])                      # log-odds of P(y = 1)
            loss = np.sum(np.logaddexp(0.0, raw) - T * raw)
            g = expit(raw) - T                              # d loss / d raw
            return loss, np.array([-g @ F, -g.sum()])

        AB0 = np.array([0.0, log((prior0 + 1.0) / (prior1 + 1.0))])
        result = minimize(loss_grad, AB0, method="L-BFGS-B", jac=True,
                          options={"gtol": 1e-6, "ftol": 64 * np.finfo(float).eps})
        self.a_, self.b_ = result.x[0] / scale, result.x[1]
        return self

    def predict(self, scores) -> np.ndarray:
        return expit(-(self.a_ * np.asarray(scores, dtype=float).ravel() + self.b_))


def make_calibrator(method: str):
    """An unfitted calibrator for ``method`` ('sigmoid' or 'isotonic')."""
    if method == "sigmoid":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip")
    raise ValueError(f"unknown calibration method {method!r}; expected one of {CALIBRATION_FAMILIES}")


def base_scores(model, X) -> np.ndarray:
    """What the calibrator sees: decision_function if available, else P(y = 1)."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).ravel()
    return np.asarray(model.predict_proba(X), dtype=float)[:, 1]


def _calibrated(calibrator, scores) -> np.ndarray:
    return np.clip(calibrator.predict(scores), 0.0, 1.0)


class Stage1CalibratedModel:
    """
    The stage's final scorer: base model fitted on the full training split,
    then the selected calibrator (fitted on out-of-fold scores).

    ``predict_proba(X)`` takes the same scaled features as the base model and
    returns ``[P(y = 0), P(y = 1)]``, like the ``CalibratedClassifierCV`` it
    replaces.
    """

    classes_ = np.array([0, 1])

    def __init__(self, base, calibrator, method: str):
        self.base, self.calibrator, self.method = base, calibrator, method

    def predict_proba(self, X) -> np.ndarray:
        p = _calibrated(self.calibrator, base_scores(self.base, X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Stage1CalibratedModel(method={self.method!r}, base={type(self.base).__name__})"


# ----------------------------------------------------------------------
# the procedure
# ----------------------------------------------------------------------
@dataclass
class OOFCalibration:
    """Everything ``calibrate_oof`` produced (indexed like the training split)."""

    method: str                                   # final model's family (step 4)
    requested: str
    calibrator: Any                               # fitted on all step-1 scores
    proba_oof: pd.Series                          # the out-of-fold output (step 3)
    fold_family: Dict[int, str]                   # outer fold -> family used for its rows
    fold_inner_brier: Dict[int, Dict[str, float]]  # the inner Brier each choice was made on
    calibrated_oof: Dict[str, pd.Series]          # per family, step 2
    base_scores_oof: pd.Series                    # step 1
    base_proba_oof: pd.Series                     # uncalibrated P(y = 1), out of fold
    brier_oof: Dict[str, float]
    provenance: str
    metrics: Dict[str, Any] = field(default_factory=dict)


def _fit_on(base, design: FoldDesign):
    """A fresh clone of ``base`` fitted on one fold's training matrix."""
    return clone(base).fit(design.X_train, design.y_train)


def _choose(brier: Dict[str, float]) -> str:
    """Lower Brier wins; ties go to sigmoid."""
    return min(brier, key=lambda f: (brier[f], CALIBRATION_FAMILIES.index(f)))


def _inner_brier(s_inner, y_train, inner_fold, families) -> Dict[str, float]:
    """Each family's calibrator cross-fitted over the inner folds, scored by Brier."""
    brier = {}
    for f in families:
        p = np.full(len(s_inner), np.nan)
        for j in np.unique(inner_fold):
            fit, held = inner_fold != j, inner_fold == j
            p[held] = _calibrated(make_calibrator(f).fit(s_inner[fit], y_train[fit]), s_inner[held])
        brier[f] = float(brier_score_loss(y_train, p))
    return brier


def _outer_fold(base, design, X, y, X_raw, cv_folds, random_state, families, make_engineer):
    """Steps 1-3 for one outer fold; touches only its training rows."""
    model = _fit_on(base, design)
    p_val = np.asarray(model.predict_proba(design.X_val), dtype=float)[:, 1]
    s_val = (np.asarray(model.decision_function(design.X_val), dtype=float).ravel()
             if hasattr(model, "decision_function") else p_val)      # = base_scores, one call

    inner = inner_fold_designs(design, X, y, cv_folds=cv_folds, random_state=random_state,
                               X_raw=X_raw, make_engineer=make_engineer)
    s_inner = np.full(len(design.train), np.nan)
    inner_fold = np.full(len(design.train), -1)
    for d in inner:
        s_inner[d.val] = base_scores(_fit_on(base, d), d.X_val)
        inner_fold[d.val] = d.fold
    if np.isnan(s_inner).any():
        raise AssertionError(f"outer fold {design.fold}: inner folds left rows unscored")

    calibrated = {f: _calibrated(make_calibrator(f).fit(s_inner, design.y_train), s_val)
                  for f in families}
    inner_brier = (_inner_brier(s_inner, design.y_train, inner_fold, families)
                   if len(families) > 1 else {})
    family = _choose(inner_brier) if inner_brier else families[0]
    return design.fold, design.val, s_val, p_val, calibrated, family, inner_brier


def calibrate_oof(
    base,
    designs: List[FoldDesign],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv_folds: int,
    random_state: int,
    method: str = "auto",
    X_raw: Optional[pd.DataFrame] = None,
    make_engineer: Optional[Callable[[], Any]] = None,
    n_jobs: Optional[int] = -1,
    strict: bool = True,
) -> OOFCalibration:
    """
    Run the procedure in the module docstring.

    Parameters
    ----------
    base : estimator
        Template for every fold model (cloned, never fitted here). The stage's
        full-train model is fine: ``clone`` drops its fit.
    designs : list of FoldDesign
        The outer folds from ``build_fold_designs`` (same X / y / X_raw).
    X, y, X_raw, make_engineer
        As passed to ``build_fold_designs``; the inner folds are rebuilt from
        them, restricted to each outer fold's training rows.
    cv_folds, random_state : int
        Inner partition: ``StratifiedKFold(cv_folds, shuffle=True,
        random_state)`` over the outer fold's training rows.
    method : {'auto', 'sigmoid', 'isotonic'}
        'auto' selects by out-of-fold Brier (step 2); otherwise the family is
        fixed and only that one is computed.
    n_jobs : int, optional
        Parallelism over outer folds.
    strict : bool
        Passed to ``assert_refittable`` on ``base``.

    Returns
    -------
    OOFCalibration
    """
    if method != "auto" and method not in CALIBRATION_FAMILIES:
        raise ValueError(f"method must be 'auto' or one of {CALIBRATION_FAMILIES}, got {method!r}")
    guard = assert_refittable(base, strict=strict)        # a prefit base survives cloning
    families = CALIBRATION_FAMILIES if method == "auto" else (method,)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_outer_fold)(base, d, X, y, X_raw, cv_folds, random_state, families, make_engineer)
        for d in designs
    )

    n = len(y)
    scores, proba, selected = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    calibrated = {f: np.full(n, np.nan) for f in families}
    fold_family, fold_inner_brier = {}, {}
    for fold, val, s_val, p_val, cal, family, inner_brier in results:
        scores[val], proba[val], selected[val] = s_val, p_val, cal[family]
        for f in families:
            calibrated[f][val] = cal[f]
        fold_family[fold], fold_inner_brier[fold] = family, inner_brier
    if np.isnan(scores).any():
        raise AssertionError("some training rows received no out-of-fold score")

    y_arr = np.asarray(y)
    brier = {f: float(brier_score_loss(y_arr, calibrated[f])) for f in families}
    chosen = _choose(brier)
    calibrator = make_calibrator(chosen).fit(scores, y_arr)

    provenance = guard if guard.upper().startswith("PREFIT") else (
        f"refittable (nested cross-fit: {len(designs)} outer x {cv_folds} inner folds; "
        "encoder, scaler, model and calibrator fitted per fold)"
    )
    auto = method == "auto"
    metrics = {
        "brier_uncalibrated": float(brier_score_loss(y_arr, proba)),
        "brier_calibrated": float(brier_score_loss(y_arr, selected)),   # the OOF output
        "ece_uncalibrated": float(calculate_ece(y_arr, proba)),
        "ece_calibrated": float(calculate_ece(y_arr, selected)),
        **{f"brier_oof_{f}": brier[f] for f in families},
        "selection": "nested_oof_brier" if auto else "requested",      # final model (step 4)
        "oof_selection": "per_fold_inner_brier" if auto else "requested",  # OOF output (step 3)
        "oof_family_by_fold": ",".join(fold_family[k] for k in sorted(fold_family)),
        "metrics_population": "out_of_fold",
    }
    index = y.index
    return OOFCalibration(
        method=chosen, requested=method, calibrator=calibrator,
        proba_oof=pd.Series(selected, index=index, name="stage1_proba_oof"),
        fold_family=fold_family, fold_inner_brier=fold_inner_brier,
        calibrated_oof={f: pd.Series(calibrated[f], index=index, name="stage1_proba_oof")
                        for f in families},
        base_scores_oof=pd.Series(scores, index=index, name="stage1_base_score_oof"),
        base_proba_oof=pd.Series(proba, index=index, name="stage1_base_proba_oof"),
        brier_oof=brier, provenance=provenance, metrics=metrics,
    )
