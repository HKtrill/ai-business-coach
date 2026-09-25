"""
blackbox_pipeline.models.xgb.shared_protocol
============================================
Stage 3 XGBoost on the SHARED Stage 3 protocol (``shared.stage_runner``, the
module ``glass_pipeline.ebm.config`` inherits from) — the same
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

    from shared.stage_runner import Stage3Block
    STAGE3_BLOCK = Stage3Block.from_global_split(GLOBAL_SPLIT, EBMFeaturePipeline,
                                                 XGB_STAGE3_CONFIG, split_fingerprint=_FP_NOW)
    XGB_STAGE3, XGB_STAGE3_PATH = train_xgb_stage(block=STAGE3_BLOCK, config=XGB_STAGE3_CONFIG)

Supersedes the standalone pre-``shared.stage3`` implementation in this
package (``config`` / ``stage`` / ``folds`` / ``oof`` / ...), which fitted
features once on all of train and cannot feed the audited Stage 4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd

# Same module as glass_pipeline.ebm.config, so both arms run one implementation.
from shared.stage_runner import Stage3ArtifactStore, Stage3Config, Stage3Runner

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
        out = dict(params)
        for k in _INT_PARAMS:
            if k in out:
                out[k] = int(out[k])
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
    """Shared Stage3Runner with the XGBoost estimator → (Stage3Artifact, path)."""
    cfg = config or XGBStage3Config()
    if not isinstance(cfg, Stage3Config):
        raise TypeError("config must be a shared.stage_runner.Stage3Config (XGBStage3Config)")
    runner = Stage3Runner(cfg, XGBEstimator(cfg.search_space, cfg.random_state, cfg.n_jobs))
    artifact = runner.fit(block, params=params, params_source=params_source,
                          params_provenance=params_provenance, extras=extras)
    path = runner.save(verbose=cfg.verbose) if save else None
    return artifact, path
