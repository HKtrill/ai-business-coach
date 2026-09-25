"""
shared.stage3.estimator
=======================
The estimator interface each arm implements (``Stage3Estimator``), plus the
model wrappers that apply the fitted calibrator: ``CalibratedModel`` (on
engineered features) and ``Stage3ServingModel`` (on raw rows).
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .calibration import Stage3Calibrator


@runtime_checkable
class Stage3Estimator(Protocol):
    """What an arm must provide."""

    family: str
    as_array: bool          # True → models are fitted on float ndarrays

    def suggest(self, trial) -> dict[str, Any]: ...
    def normalize_params(self, params: dict[str, Any]) -> dict[str, Any]: ...
    def fit(self, params: dict[str, Any], X: pd.DataFrame, y,
            sample_weight: Optional[np.ndarray] = None): ...
    def positive_proba(self, model, X: pd.DataFrame) -> np.ndarray: ...
    def search_space_dict(self) -> dict: ...
    def describe_space(self) -> dict[str, str]: ...
    def to_dict(self) -> dict: ...


def positive_proba(model, X, as_array: bool) -> np.ndarray:
    """``P(y = 1 | x)``, robust to a single-class fold."""
    data = np.asarray(X, dtype=float) if as_array else X
    proba = model.predict_proba(data)
    classes = list(getattr(model, "classes_", [0, 1]))
    if proba.shape[1] == 1:
        return np.full(len(X), float(classes[0]), dtype=float)
    return np.asarray(proba[:, classes.index(1)], dtype=float)


def _apply_calibrator(calibrator, p: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return p.copy()
    return Stage3Calibrator._apply(calibrator, p)


class CalibratedModel:
    """Refit model + OOF-fitted calibrator, called on ENGINEERED features."""

    def __init__(self, model, calibrator, as_array: bool, features: list[str]):
        self.model = model
        self.calibrator = calibrator
        self.as_array = bool(as_array)
        self.features = list(features)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features] if isinstance(X, pd.DataFrame) else X
        p = _apply_calibrator(
            self.calibrator, positive_proba(self.model, X, self.as_array)
        )
        return np.column_stack([1.0 - p, p])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class Stage3ServingModel:
    """Raw GLOBAL_SPLIT-shaped rows → Stage 3 probabilities."""

    def __init__(self, pipeline, cleaner, model, calibrator,
                 as_array: bool, features: list[str]):
        self.pipeline = pipeline
        self.cleaner = cleaner
        self.model = model
        self.calibrator = calibrator
        self.as_array = bool(as_array)
        self.features = list(features)

    def engineer(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X = self.pipeline.transform(X_raw.copy())
        return self.cleaner.transform(X[self.features])

    def predict_proba_raw(self, X_raw: pd.DataFrame) -> np.ndarray:
        return positive_proba(self.model, self.engineer(X_raw), self.as_array)

    def predict_proba_calibrated(self, X_raw: pd.DataFrame) -> np.ndarray:
        return _apply_calibrator(self.calibrator, self.predict_proba_raw(X_raw))
