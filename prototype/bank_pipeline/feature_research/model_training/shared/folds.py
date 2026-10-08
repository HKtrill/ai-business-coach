"""
model_training.folds
====================
Leakage-safe cross-validation plumbing shared by the stage trainers.

build_folds() splits the training rows with the caller's splitter. When a
feature factory is given, a fresh feature pipeline is fit on each fold's
training rows and only transforms its validation rows, so target-derived
features never see validation labels. Model columns are selected and checked
per fold. engineer_full() applies the same transform to all rows for the final
fit.

Public API
----------
Fold                                                   one fold's model-ready data
FeatureFactory                                         zero-arg callable → transformer
build_folds(X, y, features, cv, factory, label, check) → list[Fold]
engineer_full(X, y, features, factory, check)          → DataFrame
require_finite(X, where)                               numeric + finite check
recall_at_fpr(y_true, y_score, max_fpr)                → float
factory_name(factory)                                  → str (console label)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import BaseCrossValidator

# Zero-arg callable returning an unfitted transformer with fit_transform(X, y)
# and transform(X), e.g. partial(FeaturePipeline, **FE_PARAMS).
FeatureFactory = Callable[[], Any]

# Column check applied to every model-ready matrix: check(X, where) raises on failure.
ColumnCheck = Callable[[pd.DataFrame, str], None]


@dataclass(frozen=True)
class Fold:
    """One CV fold's model-ready matrices; va_idx = positional rows of the validation part."""
    X_tr: pd.DataFrame
    X_va: pd.DataFrame
    y_tr: pd.Series
    y_va: pd.Series
    va_idx: np.ndarray


def require_finite(X: pd.DataFrame, where: str) -> None:
    """Raise if any column is non-numeric (TypeError) or non-finite (ValueError). Nothing is imputed."""
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise TypeError(f"{where}: non-numeric features {non_numeric}")
    finite = np.isfinite(X.to_numpy(dtype=float))
    bad = X.columns[~finite.all(axis=0)].tolist()
    if bad:
        raise ValueError(f"{where}: non-finite values in {bad}. Fix upstream — trainers do not impute.")


def _select(X: pd.DataFrame, features: list[str], where: str, check: ColumnCheck) -> pd.DataFrame:
    """Return X[features] in that order; raise if any are missing or fail `check`."""
    missing = [f for f in features if f not in X.columns]
    if missing:
        raise KeyError(f"{where}: missing features {missing}")
    out = X.loc[:, list(features)]
    check(out, where)
    return out


def build_folds(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    cv: BaseCrossValidator,
    feature_factory: Optional[FeatureFactory] = None,
    label: str = "cv",
    check: ColumnCheck = require_finite,
) -> list[Fold]:
    """
    Split with `cv`; if a factory is given, fit a fresh feature pipeline on the
    fold's training rows and transform its validation rows; select `features`
    and run `check` on both parts. Build once per CV and reuse across trials.
    """
    if not X.index.equals(y.index):
        raise ValueError("X and y indices are not aligned.")
    folds: list[Fold] = []
    for k, (tr, va) in enumerate(cv.split(X, y), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        if feature_factory is not None:
            fe = feature_factory()
            X_tr = fe.fit_transform(X_tr, y_tr)
            X_va = fe.transform(X_va)
        folds.append(Fold(
            X_tr=_select(X_tr, features, f"{label} fold {k} train", check),
            X_va=_select(X_va, features, f"{label} fold {k} val", check),
            y_tr=y_tr,
            y_va=y_va,
            va_idx=va,
        ))
    return folds


def engineer_full(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    feature_factory: Optional[FeatureFactory] = None,
    check: ColumnCheck = require_finite,
) -> pd.DataFrame:
    """Model-ready matrix for all input rows (final fit only)."""
    if feature_factory is not None:
        X = feature_factory().fit_transform(X, y)
    return _select(X, features, "full fit", check)


def recall_at_fpr(y_true, y_score, max_fpr: float = 0.10) -> float:
    """Highest TPR on the ROC curve with FPR ≤ max_fpr (never exceeds the FPR budget)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(tpr[fpr <= max_fpr].max())


def _callable_name(f: Any) -> str:
    """Short name for a factory; unwraps functools.partial (e.g. BinaryFeaturePipeline(FeaturePipeline))."""
    if isinstance(f, partial):
        inner = ", ".join(_callable_name(a) for a in f.args if callable(a))
        return f"{_callable_name(f.func)}({inner})" if inner else _callable_name(f.func)
    return getattr(f, "__name__", type(f).__name__)


def factory_name(feature_factory: Optional[FeatureFactory]) -> str:
    """Readable description of the feature step for console reports."""
    if feature_factory is None:
        return "pre-engineered input (no per-fold refit)"
    return f"refit per fold — {_callable_name(feature_factory)}"
