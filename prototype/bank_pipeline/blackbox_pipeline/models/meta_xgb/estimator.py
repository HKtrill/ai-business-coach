"""
blackbox_pipeline.models.meta_xgb.estimator
===========================================
``MetaXGBEstimator`` — the ``Stage3Estimator`` interface for the Stage 4
meta-learner, so the shared runner can tune / cross-fit / refit it.

Missing values (the router probability on abstain rows) are handled natively
by XGBoost's default-direction splits; nothing is imputed.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import MetaXGBSearchSpace

_INT_PARAMS = ("n_estimators", "max_depth")

#: The tuned hyperparameters, in ``suggest`` order. ``colsample_bytree`` is a
#: fixed field of the search space, applied by ``_model``; it is not tuned.
META_XGB_TUNED_PARAM_NAMES: tuple[str, ...] = (
    "n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample",
    "reg_lambda", "reg_alpha",
)

#: XGBoost's own domain per tuned value, as (lower bound, lower inclusive,
#: upper bound). XGBoost accepts several out-of-domain values without a word:
#: n_estimators <= 0 (no trees), learning_rate 0 or NaN, subsample 0.
_META_XGB_DOMAIN: dict[str, tuple[float, bool, float]] = {
    "n_estimators": (1, True, math.inf),
    "learning_rate": (0.0, False, math.inf),
    "max_depth": (0, True, math.inf),
    "min_child_weight": (0.0, True, math.inf),
    "subsample": (0.0, False, 1.0),
    "reg_lambda": (0.0, True, math.inf),
    "reg_alpha": (0.0, True, math.inf),
}


class MetaXGBEstimator:
    family = "meta_xgb"
    as_array = False

    def __init__(self, space: Optional[MetaXGBSearchSpace] = None,
                 random_state: int = 42, n_jobs: int = 1):
        self.space = space or MetaXGBSearchSpace()
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)

    # ------------------------------------------------------------------
    def suggest(self, trial) -> dict[str, Any]:
        s = self.space
        lo, hi, step = s.n_estimators
        return {
            "n_estimators": trial.suggest_int("n_estimators", lo, hi, step=step),
            "learning_rate": trial.suggest_float("learning_rate", *s.learning_rate, log=True),
            "max_depth": trial.suggest_int("max_depth", *s.max_depth),
            "min_child_weight": trial.suggest_float("min_child_weight", *s.min_child_weight, log=True),
            "subsample": trial.suggest_float("subsample", *s.subsample),
            "reg_lambda": trial.suggest_float("reg_lambda", *s.reg_lambda, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", *s.reg_alpha, log=True),
        }

    def normalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Exactly the tuned keys, each a finite number in XGBoost's domain.

        PR 36 (audit F1): params used to reach ``XGBClassifier`` as given. At
        ``verbosity=0`` XGBoost ignores an unknown or misspelled name without a
        word (``max_dept=4`` was recorded in ``best_params`` and never
        applied), a missing key silently took XGBoost's default (``max_depth``
        6, outside the space's 1-4), and ``int()`` truncated
        ``n_estimators=20.7``. Now, as in the Stage 3 ``XGBEstimator``, a
        missing or unknown key (aliases such as ``eta`` and the fixed
        ``colsample_bytree`` / ``missing`` included) raises ``KeyError``; a
        value that is not a finite real number, a non-integral
        ``n_estimators`` / ``max_depth``, or a value outside XGBoost's domain
        raises ``ValueError``.
        """
        missing = [k for k in META_XGB_TUNED_PARAM_NAMES if k not in params]
        if missing:
            raise KeyError(f"Meta-XGB params missing tuned keys: {missing}")
        unknown = [k for k in params if k not in META_XGB_TUNED_PARAM_NAMES]
        if unknown:
            raise KeyError(f"Unknown Meta-XGB hyperparameters {unknown}; expected exactly "
                           f"{list(META_XGB_TUNED_PARAM_NAMES)}.")
        out: dict[str, Any] = {}
        for name in META_XGB_TUNED_PARAM_NAMES:
            value = params[name]
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                    or not math.isfinite(value)):
                raise ValueError(f"Meta-XGB {name} must be a finite number, got {value!r}")
            if name in _INT_PARAMS:
                if not float(value).is_integer():
                    raise ValueError(f"Meta-XGB {name} must be an integer, got {value!r}")
                value = int(value)
            else:
                value = float(value)
            low, low_inclusive, high = _META_XGB_DOMAIN[name]
            if not ((value >= low) if low_inclusive else (value > low)) or value > high:
                raise ValueError(f"Meta-XGB {name}={value!r} is outside its domain "
                                 f"{'[' if low_inclusive else '('}{low}, {high}]")
            out[name] = value
        return out

    def _model(self, params: dict[str, Any]):
        from xgboost import XGBClassifier
        return XGBClassifier(
            **self.normalize_params(params),
            colsample_bytree=self.space.colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            missing=np.nan,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0,
        )

    def fit(self, params: dict[str, Any], X: pd.DataFrame, y,
            sample_weight: Optional[np.ndarray] = None):
        model = self._model(params)
        model.fit(X, np.asarray(y).astype(int), sample_weight=sample_weight)
        return model

    def positive_proba(self, model, X: pd.DataFrame) -> np.ndarray:
        proba = model.predict_proba(X)
        classes = list(getattr(model, "classes_", [0, 1]))
        if proba.shape[1] == 1:
            return np.full(len(X), float(classes[0]), dtype=float)
        return np.asarray(proba[:, classes.index(1)], dtype=float)

    # ------------------------------------------------------------------
    def search_space_dict(self) -> dict:
        return self.space.to_dict()

    def describe_space(self) -> dict[str, str]:
        s = self.space
        return {
            "n_estimators": f"int[{s.n_estimators[0]}, {s.n_estimators[1]}] step {s.n_estimators[2]}",
            "learning_rate": f"float[{s.learning_rate[0]}, {s.learning_rate[1]}] log",
            "max_depth": f"int[{s.max_depth[0]}, {s.max_depth[1]}]",
            "min_child_weight": f"float[{s.min_child_weight[0]}, {s.min_child_weight[1]}] log",
            "subsample": f"float[{s.subsample[0]}, {s.subsample[1]}]",
            "reg_lambda": f"float[{s.reg_lambda[0]}, {s.reg_lambda[1]}] log",
            "reg_alpha": f"float[{s.reg_alpha[0]}, {s.reg_alpha[1]}] log",
            "colsample_bytree": f"fixed {s.colsample_bytree}",
        }

    def to_dict(self) -> dict:
        return {"family": self.family, "search_space": self.search_space_dict(),
                "random_state": self.random_state, "n_jobs": self.n_jobs,
                "fixed": {"objective": "binary:logistic", "tree_method": "hist",
                          "missing": "nan (native)", "eval_metric": "logloss"}}
