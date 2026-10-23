"""
blackbox_pipeline.models.xgb.shared_protocol
============================================
Stage 3 XGBoost on the SHARED Stage 3 protocol (``shared.stage3``, the
package ``glass_pipeline.ebm.config`` inherits from) — the same
runner, block, folds, weights, tuning, OOF, calibration, thresholds and
artifact the GLASS EBM uses. Only two things are XGBoost-specific:

* ``XGBEstimator``   — builds / fits ``XGBClassifier`` (the Stage3Estimator
                       interface the shared runner calls);
* ``XGBSearchSpace`` — the XGBoost-native Optuna space (bounds unchanged from
                       the previous package: see ``describe_space``).

Everything else comes from ``Stage3Config`` defaults, so
``shared.stage3.assert_matched_protocol(EBM_STAGE3, XGB_STAGE3)`` compares like
with like. If the GLASS EBM config overrides a shared default, that check
names the field — align it here rather than weakening the check.

    from shared.stage3 import Stage3Block
    STAGE3_BLOCK = Stage3Block.from_global_split(GLOBAL_SPLIT, EBMFeaturePipeline,
                                                 XGB_STAGE3_CONFIG, split_fingerprint=_FP_NOW)
    XGB_STAGE3, XGB_STAGE3_PATH = train_xgb_stage(block=STAGE3_BLOCK, config=XGB_STAGE3_CONFIG)

Superseded the standalone pre-``shared.stage3`` implementation that used to
live in this package (``config`` / ``stage`` / ``folds`` / ``oof`` / ...),
which fitted features once on all of train and could not feed the audited
Stage 4; it was removed in PR 36 Commit 4.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd

# Same module as glass_pipeline.ebm.config, so both arms run one implementation.
from shared.stage3.config import Stage3Config
from shared.stage3.runner import Stage3Runner

_INT_PARAMS = ("n_estimators", "max_depth")


@dataclass
class XGBSearchSpace:
    """XGBoost-native space; bounds as in the previous tuning runs."""

    n_estimators: tuple = (200, 2000, 100)       # low, high, step
    learning_rate: tuple = (0.005, 0.05)         # log
    max_depth: tuple = (2, 8)
    min_child_weight: tuple = (1.0, 20.0)        # log
    subsample: tuple = (0.6, 1.0)
    colsample_bytree: tuple = (0.6, 1.0)
    reg_lambda: tuple = (0.001, 10.0)            # log

    def to_dict(self) -> dict:
        return asdict(self)


#: The tuned hyperparameters, in search-space (and ``suggest``) order.
XGB_TUNED_PARAM_NAMES: tuple[str, ...] = tuple(f.name for f in fields(XGBSearchSpace))

#: XGBoost's own domain per tuned value, as (lower bound, lower inclusive,
#: upper bound). Enforced because XGBoost accepts several out-of-domain values
#: without a word: n_estimators <= 0 (no trees), learning_rate 0 or NaN,
#: subsample / colsample_bytree 0. max_depth 0 means "no limit" and is valid.
_XGB_DOMAIN: dict[str, tuple[float, bool, float]] = {
    "n_estimators": (1, True, math.inf),
    "learning_rate": (0.0, False, math.inf),
    "max_depth": (0, True, math.inf),
    "min_child_weight": (0.0, True, math.inf),
    "subsample": (0.0, False, 1.0),
    "colsample_bytree": (0.0, False, 1.0),
    "reg_lambda": (0.0, True, math.inf),
}


@dataclass
class XGBStage3Config(Stage3Config):
    """Shared Stage 3 protocol; XGBoost identity + search space only."""

    model_family: str = "xgboost"
    arm: str = "blackbox"
    counterpart: str = "glass_pipeline.ebm"
    column_prefix: str = "xgb_stage3_"
    study_name: str = "xgb_stage3_recall_biased"
    search_space: Any = field(default_factory=XGBSearchSpace)
    artifact_dir: str = "models/xgb"
    artifact_stem: str = "xgb_stage3"

    _space_cls: ClassVar[Optional[type]] = XGBSearchSpace


class XGBEstimator:
    """``Stage3Estimator`` for XGBoost."""

    family = "xgboost"
    as_array = False

    def __init__(self, space: Optional[XGBSearchSpace] = None,
                 random_state: int = 42, n_jobs: int = 1):
        self.space = space or XGBSearchSpace()
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)

    def suggest(self, trial) -> dict[str, Any]:
        s = self.space
        lo, hi, step = s.n_estimators
        return {
            "n_estimators": trial.suggest_int("n_estimators", lo, hi, step=step),
            "learning_rate": trial.suggest_float("learning_rate", *s.learning_rate, log=True),
            "max_depth": trial.suggest_int("max_depth", *s.max_depth),
            "min_child_weight": trial.suggest_float("min_child_weight", *s.min_child_weight, log=True),
            "subsample": trial.suggest_float("subsample", *s.subsample),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *s.colsample_bytree),
            "reg_lambda": trial.suggest_float("reg_lambda", *s.reg_lambda, log=True),
        }

    def normalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Exactly the tuned keys, each a finite number in XGBoost's domain.

        PR 36 (audit F1): params used to reach ``XGBClassifier`` as given. At
        ``verbosity=0`` XGBoost ignores an unknown or misspelled name without a
        word (``max_dept=4`` trained at the default depth while the artifact
        recorded ``max_dept=4``), a missing key silently took XGBoost's
        default, and ``int()`` truncated ``n_estimators=20.7``. Now, as in
        ``EBMFactory.normalize_params``, a missing or unknown key (aliases such
        as ``eta`` and fixed conditions such as ``random_state`` included)
        raises ``KeyError``; a value that is not a finite real number, a
        non-integral ``n_estimators`` / ``max_depth``, or a value outside
        XGBoost's domain raises ``ValueError``.
        """
        missing = [k for k in XGB_TUNED_PARAM_NAMES if k not in params]
        if missing:
            raise KeyError(f"XGBoost params missing tuned keys: {missing}")
        unknown = [k for k in params if k not in XGB_TUNED_PARAM_NAMES]
        if unknown:
            raise KeyError(f"Unknown XGBoost hyperparameters {unknown}; expected exactly "
                           f"{list(XGB_TUNED_PARAM_NAMES)}.")
        out: dict[str, Any] = {}
        for name in XGB_TUNED_PARAM_NAMES:
            value = params[name]
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                    or not math.isfinite(value)):
                raise ValueError(f"XGBoost {name} must be a finite number, got {value!r}")
            if name in _INT_PARAMS:
                if not float(value).is_integer():
                    raise ValueError(f"XGBoost {name} must be an integer, got {value!r}")
                value = int(value)
            else:
                value = float(value)
            low, low_inclusive, high = _XGB_DOMAIN[name]
            if not ((value >= low) if low_inclusive else (value > low)) or value > high:
                raise ValueError(f"XGBoost {name}={value!r} is outside its domain "
                                 f"{'[' if low_inclusive else '('}{low}, {high}]")
            out[name] = value
        return out

    def fit(self, params: dict[str, Any], X: pd.DataFrame, y,
            sample_weight: Optional[np.ndarray] = None):
        from xgboost import XGBClassifier
        model = XGBClassifier(
            **self.normalize_params(params),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0,
        )
        model.fit(X, np.asarray(y).astype(int), sample_weight=sample_weight)
        return model

    def positive_proba(self, model, X: pd.DataFrame) -> np.ndarray:
        proba = model.predict_proba(X)
        classes = list(getattr(model, "classes_", [0, 1]))
        if proba.shape[1] == 1:
            return np.full(len(X), float(classes[0]), dtype=float)
        return np.asarray(proba[:, classes.index(1)], dtype=float)

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
            "colsample_bytree": f"float[{s.colsample_bytree[0]}, {s.colsample_bytree[1]}]",
            "reg_lambda": f"float[{s.reg_lambda[0]}, {s.reg_lambda[1]}] log",
        }

    def to_dict(self) -> dict:
        return {"family": self.family, "search_space": self.search_space_dict(),
                "random_state": self.random_state, "n_jobs": self.n_jobs,
                "fixed": {"objective": "binary:logistic", "tree_method": "hist",
                          "eval_metric": "logloss"}}


def train_xgb_stage(
    block,
    config: Optional[XGBStage3Config] = None,
    params: Optional[dict] = None,
    params_source: Optional[str] = None,
    params_provenance: Optional[dict] = None,
    extras: Optional[dict] = None,
    save: bool = True,
):
    """
    Shared Stage3Runner with the XGBoost estimator → (Stage3Artifact, path).

    PR 36 (audit F37): the config alone labels the artifact (model family,
    arm, column prefix, search space), and any ``Stage3Config`` used to be
    accepted. An ``EBMStage3Config`` therefore produced an XGBoost model
    labelled ``model_family='ebm'``, ``arm='glass'`` — the identity the GLASS
    arbiter's F32 check accepts — and the generic config labelled it
    ``'unset'``. Only an ``XGBStage3Config`` with the black-box XGBoost
    identity is accepted.
    """
    cfg = config or XGBStage3Config()
    if not isinstance(cfg, XGBStage3Config):
        raise TypeError(
            f"config must be an XGBStage3Config, got {type(cfg).__name__}: the config "
            "labels the artifact, and Stage 4 checks that label (audit F37).")
    if (cfg.model_family, cfg.arm) != (XGBEstimator.family, "blackbox"):
        raise ValueError(
            "XGBStage3Config must keep the black-box XGBoost identity "
            f"(model_family='xgboost', arm='blackbox'), got model_family="
            f"{cfg.model_family!r}, arm={cfg.arm!r} (audit F37).")
    runner = Stage3Runner(cfg, XGBEstimator(cfg.search_space, cfg.random_state, cfg.n_jobs))
    artifact = runner.fit(block, params=params, params_source=params_source,
                          params_provenance=params_provenance, extras=extras)
    path = runner.save(verbose=cfg.verbose) if save else None
    return artifact, path
