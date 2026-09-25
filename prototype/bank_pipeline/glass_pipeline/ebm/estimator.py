"""
glass_pipeline.ebm.estimator
============================
The EBM arm's implementation of ``shared.stage_runner.Stage3Estimator``.

``EBMFactory`` is the single place an ``ExplainableBoostingClassifier`` is
built — Optuna objective, 10-fold report, OOF pass and full refit all go
through it (the old ``_build_model`` discipline). Its counterpart in the
black-box arm is ``blackbox_pipeline.models.xgb.estimator.XGBFactory``.

Search space (unchanged from the research trainer)
--------------------------------------------------
    learning_rate         float[0.005, 0.05] log
    max_rounds            int[500, 5000] step 100
    max_bins              int[128, 512] step 32
    max_interaction_bins  int[16, 128] step 16

Parameters are suggested in that order so the TPE sequence under seed 42 is
the same as before PR 33. The validated interaction pairs are fixed, not
searched. ``n_jobs=1`` throughout.

EBMs are fitted on DataFrames (``as_array = False``): the interaction pairs
are declared by column name.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

__all__ = ["EBMSearchSpace", "EBMFactory", "EBM_TUNED_PARAM_NAMES"]

EBM_TUNED_PARAM_NAMES: tuple[str, ...] = (
    "learning_rate",
    "max_rounds",
    "max_bins",
    "max_interaction_bins",
)


@dataclass(frozen=True)
class EBMSearchSpace:
    learning_rate_low: float = 0.005
    learning_rate_high: float = 0.05
    learning_rate_log: bool = True

    max_rounds_low: int = 500
    max_rounds_high: int = 5000
    max_rounds_step: int = 100

    max_bins_low: int = 128
    max_bins_high: int = 512
    max_bins_step: int = 32

    max_interaction_bins_low: int = 16
    max_interaction_bins_high: int = 128
    max_interaction_bins_step: int = 16

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> dict[str, str]:
        return {
            "learning_rate": f"float[{self.learning_rate_low}, "
                             f"{self.learning_rate_high}] "
                             f"{'log' if self.learning_rate_log else 'linear'}",
            "max_rounds": f"int[{self.max_rounds_low}, {self.max_rounds_high}] "
                          f"step {self.max_rounds_step}",
            "max_bins": f"int[{self.max_bins_low}, {self.max_bins_high}] "
                        f"step {self.max_bins_step}",
            "max_interaction_bins": f"int[{self.max_interaction_bins_low}, "
                                    f"{self.max_interaction_bins_high}] "
                                    f"step {self.max_interaction_bins_step}",
        }


class EBMFactory:
    family = "ebm"
    as_array = False

    def __init__(
        self,
        interactions: list[tuple[str, str]],
        search_space: Optional[EBMSearchSpace] = None,
        random_state: int = 42,
        n_jobs: int = 1,
        extra_params: Optional[dict[str, Any]] = None,
    ):
        self.interactions = [tuple(p) for p in interactions]
        self.search_space = search_space or EBMSearchSpace()
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.extra_params = dict(extra_params or {})

    # ------------------------------------------------------------------
    def suggest(self, trial) -> dict[str, Any]:
        s = self.search_space
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", s.learning_rate_low, s.learning_rate_high,
                log=s.learning_rate_log),
            "max_rounds": trial.suggest_int(
                "max_rounds", s.max_rounds_low, s.max_rounds_high,
                step=s.max_rounds_step),
            "max_bins": trial.suggest_int(
                "max_bins", s.max_bins_low, s.max_bins_high,
                step=s.max_bins_step),
            "max_interaction_bins": trial.suggest_int(
                "max_interaction_bins", s.max_interaction_bins_low,
                s.max_interaction_bins_high, step=s.max_interaction_bins_step),
        }

    @staticmethod
    def normalize_params(trial_params: dict[str, Any]) -> dict[str, Any]:
        """Exactly the tuned keys; unknown keys raise (never silently dropped)."""
        missing = [p for p in EBM_TUNED_PARAM_NAMES if p not in trial_params]
        if missing:
            raise KeyError(f"EBM params missing canonical keys: {missing}")
        unknown = [p for p in trial_params if p not in EBM_TUNED_PARAM_NAMES]
        if unknown:
            raise KeyError(f"Unknown EBM hyperparameters {unknown}; expected "
                           f"exactly {list(EBM_TUNED_PARAM_NAMES)}.")
        return {
            "learning_rate": float(trial_params["learning_rate"]),
            "max_rounds": int(trial_params["max_rounds"]),
            "max_bins": int(trial_params["max_bins"]),
            "max_interaction_bins": int(trial_params["max_interaction_bins"]),
        }

    def search_space_dict(self) -> dict:
        return self.search_space.to_dict()

    def describe_space(self) -> dict[str, str]:
        return self.search_space.describe()

    # ------------------------------------------------------------------
    def build(self, params: dict[str, Any]) -> ExplainableBoostingClassifier:
        return ExplainableBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_rounds=params["max_rounds"],
            max_bins=params["max_bins"],
            max_interaction_bins=params["max_interaction_bins"],
            interactions=list(self.interactions),
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self.extra_params,
        )

    def fit(self, params, X: pd.DataFrame, y, sample_weight=None):
        model = self.build(params)
        model.fit(X, np.asarray(y).astype(int), sample_weight=sample_weight)
        return model

    @staticmethod
    def positive_proba(model, X: pd.DataFrame) -> np.ndarray:
        proba = model.predict_proba(X)
        classes = list(getattr(model, "classes_", [0, 1]))
        if proba.shape[1] == 1:
            return np.full(len(X), float(classes[0]), dtype=float)
        return np.asarray(proba[:, classes.index(1)], dtype=float)

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "interactions": [list(p) for p in self.interactions],
            "extra_params": self.extra_params,
            "tuned_param_names": list(EBM_TUNED_PARAM_NAMES),
            "search_space": self.search_space.to_dict(),
        }
