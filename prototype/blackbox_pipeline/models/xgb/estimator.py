"""
blackbox_pipeline.models.xgb.estimator
=======================================
Model construction.

``XGBFactory`` is the single place an ``XGBClassifier`` is built. Every phase of
the stage — the Optuna objective, the fold-wise CV report, the OOF pass and the
final refit — goes through it, so no phase can silently differ in a fixed
hyperparameter. This mirrors the EBM's ``_build_model`` discipline.

The factory holds no fitted state; it is safe to reuse across folds.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

__all__ = ["XGBFactory", "TUNED_PARAM_NAMES"]

# The canonical tuned parameters, in the order they are reported. Anything not
# on this list is a fixed condition of the experiment, not a searched value.
TUNED_PARAM_NAMES: tuple[str, ...] = (
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
)


class XGBFactory:
    """
    Builds ``XGBClassifier`` instances from canonical parameters.

    Fixed conditions (not tuned, recorded in the artifact)
    ------------------------------------------------------
    ``objective="binary:logistic"``
        Stage 3 is binary probability estimation, matching the EBM.
    ``eval_metric="logloss"``
        Reporting only. The selection objective is F2, applied outside the
        booster — the EBM likewise optimises its own internal loss while Optuna
        selects on F2.
    ``tree_method="hist"``
        Deterministic given the seed and thread count.
    ``n_jobs``
        Taken from config; defaults to 1, mirroring the EBM's ``n_jobs=1``.
        Changing it can perturb results at floating-point level, so exact
        reproduction requires the same value.

    Inputs are cast to ``float`` arrays: features must be numeric, and column
    order is whatever the caller passes (the stage enforces the contract order).
    """

    def __init__(
        self,
        random_state: int = 42,
        n_jobs: int = 1,
        tree_method: str = "hist",
        extra_params: Optional[dict[str, Any]] = None,
    ):
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.tree_method = tree_method
        self.extra_params = dict(extra_params or {})

    # ------------------------------------------------------------------
    @staticmethod
    def normalize_params(trial_params: dict[str, Any]) -> dict[str, Any]:
        """
        Collapse Optuna trial params into canonical constructor params.

        Mirrors the EBM's ``_normalize_params``: the artifact records a clean
        dict rather than whatever shape the sampler happened to produce.

        Requires exactly the keys in ``TUNED_PARAM_NAMES``; ints for
        ``n_estimators`` / ``max_depth``, floats otherwise. Unknown keys raise
        rather than being dropped, so a supplied ``params`` dict can never
        record settings that were silently not applied. Fixed conditions go
        through ``XGBFactory(extra_params=...)``.
        """
        missing = [p for p in TUNED_PARAM_NAMES if p not in trial_params]
        if missing:
            raise KeyError(f"Trial params missing canonical keys: {missing}")
        unknown = [p for p in trial_params if p not in TUNED_PARAM_NAMES]
        if unknown:
            raise KeyError(
                f"Unknown hyperparameters {unknown}; expected exactly "
                f"{list(TUNED_PARAM_NAMES)}."
            )
        out: dict[str, Any] = {}
        for name in TUNED_PARAM_NAMES:
            value = trial_params[name]
            if name in ("n_estimators", "max_depth"):
                out[name] = int(value)
            else:
                out[name] = float(value)
        return out

    # ------------------------------------------------------------------
    def fixed_params(self) -> dict[str, Any]:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": self.tree_method,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
        params.update(self.extra_params)
        return params

    def build(self, params: dict[str, Any]) -> XGBClassifier:
        """A fresh, unfitted classifier. Never returns a shared instance."""
        return XGBClassifier(**{**self.fixed_params(), **dict(params)})

    def fit(
        self,
        params: dict[str, Any],
        X: pd.DataFrame,
        y,
        sample_weight: Optional[np.ndarray] = None,
    ) -> XGBClassifier:
        model = self.build(params)
        model.fit(np.asarray(X, dtype=float), np.asarray(y).astype(int),
                  sample_weight=sample_weight)
        return model

    # ------------------------------------------------------------------
    @staticmethod
    def positive_proba(model, X: pd.DataFrame) -> np.ndarray:
        """``P(y = 1 | x)``, robust to a single-class fold."""
        proba = model.predict_proba(np.asarray(X, dtype=float))
        classes = list(getattr(model, "classes_", [0, 1]))
        if proba.shape[1] == 1:
            return np.full(len(X), float(classes[0]), dtype=float)
        return proba[:, classes.index(1)].astype(float)

    @staticmethod
    def hard_predict(model, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Class labels at an explicit cut.

        The EBM's tuning objective scores ``model.predict(X_val)``, which is the
        0.5 cut. Making the threshold explicit here keeps that visible rather
        than hidden inside a library default.
        """
        return (XGBFactory.positive_proba(model, X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "tree_method": self.tree_method,
            "fixed_params": self.fixed_params(),
            "tuned_param_names": list(TUNED_PARAM_NAMES),
        }