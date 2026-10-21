"""
shared.stage1.crossfit
======================
Per-fold preprocessing for every Stage 1 cross-validation, shared by the
GLASS ``CalibratedLRStage`` and the black-box ``CalibratedStage1MLP``.

Each arm cross-validates twice on the training split: the Optuna tuning folds
and the out-of-fold (OOF) probabilities behind the F2 threshold and the
Stage 4 train columns. A fold's held-out rows must be scored by a pipeline
that never saw them, so everything learned from the data is refit inside the
fold:

* the target encodings ``euribor3m_local_rate`` and ``dow_month_encoded``
  (audit F3). Fitted once on the whole training split, they carry each
  held-out row's own label into its features. Here ``LRFeatureEngineer`` is
  refit on the fold's training rows and applied to its held-out rows;
* the ``StandardScaler`` (audit F4), refit on the fold's training rows.

The stage's full-train fit — features from ``engineer_features``, scaler,
model, calibrator — is unchanged and remains what scores the test split.

The same builder also produces the INNER folds of each outer fold (audit
F19): ``inner_fold_designs`` refits the encoders and scaler on each inner
fold's training rows, using only the outer fold's training rows. The
calibration procedure (``shared.stage1.oof_calibration``) runs on these.

Public API
----------
TARGET_ENCODED_FEATURES
    Feature names that may only enter cross-validation through a refit.
FoldDesign
    One fold: positions, and its training / held-out matrices, preprocessed
    with statistics from its training rows only.
build_fold_designs(X, y, *, cv_folds, random_state, X_raw=None, ...)
    The folds for one Stage 1 fit, built once and reused by tuning and OOF.
inner_fold_designs(design, X, y, *, cv_folds, random_state, X_raw=None, ...)
    The same, restricted to one outer fold's training rows.
crossfit_proba(estimator, designs, n_jobs=-1)
    OOF ``P(y = 1)``: a fresh clone of ``estimator`` per fold. General helper;
    since F16/F19 the stages produce their OOF probabilities through
    ``shared.stage1.oof_calibration`` instead.
preprocessing_record(X_raw)
    What was refit per fold, for configs and artifacts.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import feature_engineering as _fe
from .feature_engineering import TARGET_ENCODED_FEATURES

__all__ = [
    "TARGET_ENCODED_FEATURES",
    "FoldDesign",
    "build_fold_designs",
    "inner_fold_designs",
    "crossfit_proba",
    "preprocessing_record",
]


@dataclass
class FoldDesign:
    """
    One Stage 1 fold.

    ``train`` / ``val`` are row positions in the rows the folds were built
    on (the training split, or an outer fold's training rows for inner
    folds); ``index_train`` / ``index_val`` the same rows as index labels.
    ``X_train`` and ``X_val`` are scaled feature matrices built from the
    fold's training rows only (encoder and scaler); ``y_train`` / ``y_val``
    the labels.
    """

    fold: int
    train: np.ndarray
    val: np.ndarray
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    index_train: Optional[pd.Index] = None
    index_val: Optional[pd.Index] = None


def _quiet_engineer(make_engineer: Optional[Callable[[], Any]], X, y):
    """Fit a fresh engineer on (X, y) without its console report."""
    make = make_engineer or _fe.LRFeatureEngineer          # looked up at call time
    with contextlib.redirect_stdout(io.StringIO()):
        return make().fit(X, y)


def _quiet_transform(engineer, X, columns) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        return engineer.transform(X)[columns]


def build_fold_designs(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv_folds: int,
    random_state: int,
    X_raw: Optional[pd.DataFrame] = None,
    make_engineer: Optional[Callable[[], Any]] = None,
) -> List[FoldDesign]:
    """
    Build the Stage 1 folds with preprocessing refit inside each fold.

    Parameters
    ----------
    X : pandas.DataFrame
        The stage's training features, as used by its full-train fit.
    y : pandas.Series
        Binary labels indexed like ``X``.
    cv_folds, random_state : int
        ``StratifiedKFold(cv_folds, shuffle=True, random_state)`` — the
        partition ``shared.stage_io.fold_assignment`` reconstructs.
    X_raw : pandas.DataFrame, optional
        The raw training rows ``X`` was engineered from (same index, same
        order). Required whenever ``X`` holds a target-encoded feature.
    make_engineer : callable, optional
        Returns an unfitted engineer with ``fit(X, y)`` / ``transform(X)``.
        Defaults to ``LRFeatureEngineer`` with its default settings — the
        engineer ``engineer_features`` uses.

    Returns
    -------
    list of FoldDesign

    Raises
    ------
    ValueError
        If ``X`` holds target encodings and ``X_raw`` is missing; if indices
        differ; or if ``X`` is not what the engineer rebuilds from ``X_raw``
        (then the folds would not replicate the full-train fit).
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X must be a DataFrame and y a Series")
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index (order included)")
    columns = list(X.columns)

    if X_raw is None:
        encoded = [c for c in columns if c in TARGET_ENCODED_FEATURES]
        if encoded:
            raise ValueError(
                f"{encoded} are target encodings fitted on the whole training split. "
                "Inside cross-validation they would carry each held-out row's own "
                "label into its score (audit F3). Pass the raw training rows they "
                "were engineered from (X_train_raw=...) so they are refit per fold."
            )
    else:
        if not X_raw.index.equals(X.index):
            raise ValueError("X_train_raw must have the same index as X_train (order included)")
        rebuilt = _quiet_transform(_quiet_engineer(make_engineer, X_raw, y), X_raw, columns)
        if not (rebuilt.index.equals(X.index)
                and np.array_equal(rebuilt.to_numpy(dtype=float), X.to_numpy(dtype=float))):
            raise ValueError(
                "X_train is not what the feature engineer produces from X_train_raw; "
                "the per-fold refits would not replicate the full-train features. "
                "Pass the raw rows and the engineer settings X_train was built with."
            )

    return _designs(X, y, X_raw, columns, cv_folds, random_state, make_engineer)


def inner_fold_designs(
    design: FoldDesign,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv_folds: int,
    random_state: int,
    X_raw: Optional[pd.DataFrame] = None,
    make_engineer: Optional[Callable[[], Any]] = None,
) -> List[FoldDesign]:
    """
    Folds over one outer fold's TRAINING rows only (audit F19).

    Same partition rule and same per-fold refits as ``build_fold_designs``,
    applied to ``design.train``; the outer held-out rows are never touched.
    ``X``, ``y`` and ``X_raw`` are the full training split, as passed to
    ``build_fold_designs``.
    """
    rows = design.train
    return _designs(
        X.iloc[rows], y.iloc[rows], X_raw.iloc[rows] if X_raw is not None else None,
        list(X.columns), cv_folds, random_state, make_engineer,
    )


def _designs(X, y, X_raw, columns, cv_folds, random_state, make_engineer) -> List[FoldDesign]:
    """Partition the given rows; refit encoder (from ``X_raw``) and scaler per fold."""
    y_arr = np.asarray(y)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    designs = []
    for k, (tr, va) in enumerate(splitter.split(np.zeros(len(y_arr)), y_arr)):
        if X_raw is None:
            F_tr, F_va = X.iloc[tr], X.iloc[va]
        else:
            engineer = _quiet_engineer(make_engineer, X_raw.iloc[tr], y.iloc[tr])
            F_tr = _quiet_transform(engineer, X_raw.iloc[tr], columns)
            F_va = _quiet_transform(engineer, X_raw.iloc[va], columns)
        scaler = StandardScaler().fit(F_tr)
        designs.append(FoldDesign(k, tr, va, scaler.transform(F_tr), scaler.transform(F_va),
                                  y_arr[tr], y_arr[va], y.index[tr], y.index[va]))
    return designs


def _fit_score(estimator, design: FoldDesign) -> np.ndarray:
    return estimator.fit(design.X_train, design.y_train).predict_proba(design.X_val)[:, 1]


def crossfit_proba(estimator, designs: List[FoldDesign], n_jobs: Optional[int] = -1) -> np.ndarray:
    """
    Out-of-fold ``P(y = 1)``: each fold scored by a fresh clone of
    ``estimator`` fitted on that fold's training matrix.

    Run :func:`shared.stage1.calibration_guard.assert_refittable` on
    ``estimator`` first: a prefit calibrator survives cloning.
    """
    n = sum(len(d.val) for d in designs)
    scores = Parallel(n_jobs=n_jobs)(delayed(_fit_score)(clone(estimator), d) for d in designs)
    proba = np.full(n, np.nan)
    for d, p in zip(designs, scores):
        proba[d.val] = p
    if np.isnan(proba).any():
        raise AssertionError("some training rows received no out-of-fold score")
    return proba


def preprocessing_record(X_raw: Optional[pd.DataFrame]) -> Dict[str, str]:
    """What each Stage 1 cross-validation refit per fold (stored in configs)."""
    return {
        "feature_fit_scope": "per_fold" if X_raw is not None else "label_free_input",
        "scaler_fit_scope": "per_fold",
        "applies_to": "optuna_tuning_folds+oof_probabilities+calibration_inner_folds",
    }
