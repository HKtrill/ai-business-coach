"""
blackbox_pipeline.models.rf_router.tuning.budget
=================================================
The tuning budget every specialization arm shares, and the router's fold plan.

Fair comparison is enforced by construction: every tuning run takes ONE
``TuningBudget`` and stamps its ``signature()`` on the result, and
``search.assert_same_budget`` refuses to compare results whose signatures or
constraints differ.

Two fold plans, on purpose
--------------------------
``RouterFolds``
    How the router builds its OOF arrays (``n_oof_folds``, ``random_state``).
    The Pass 1 remainder that Pass 2 is tuned on is built with THESE folds, so
    it is bit-for-bit the remainder the deployed router will train Pass 2 on.
``TuningBudget.fold_seed``
    How the hyperparameter search builds ITS OOF arrays. Deliberately a
    different seed from the router's, so the configuration a search selects is
    not then re-scored by the router on the very fold assignment it was selected
    on. That REDUCES the part of the selection optimism owed to fold assignment.
    It does not remove hyperparameter-selection bias: the router's OOF still
    reuses the same rows and labels the search selected on, so training-OOF
    comparisons between tuned arms stay diagnostic. Only nested tuning (tuning
    inside each outer fold) removes that dependence on the training side.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Tuple

from .space import REQUIRED_PARAM_KEYS

__all__ = ["TuningBudget", "RouterFolds"]


@dataclass(frozen=True)
class TuningBudget:
    """
    Parameters
    ----------
    n_trials
        Optuna trials per tuning run. Identical for every arm.
    sampler_seed
        TPE (or random) sampler seed. Identical for every arm.
    n_folds, fold_seed, stratify
        OOF folds the search scores each configuration on. Keep ``n_folds`` equal
        to the router's ``n_oof_folds`` so a configuration is scored on the same
        training-set size it will be deployed at; keep ``fold_seed`` different
        from the router's ``random_state`` (see module docstring).
    forest_seed
        ``random_state`` of every forest the search fits.
    fixed_params
        Normalised RF keys held constant. Default fixes ``n_estimators`` at 500:
        more trees rarely hurts a forest's ranking, so searching it spends budget
        on noise. Every re-tuned arm shares this, so it is a controlled deviation
        from the baseline search space, not an arm-specific advantage.
    forest_n_jobs
        Parallelism inside each forest. Excluded from ``signature()``: it changes
        wall clock, and probabilities only in the last ULP.
    """

    n_trials: int = 40
    sampler_seed: int = 42
    n_folds: int = 5
    fold_seed: int = 43
    forest_seed: int = 42
    stratify: bool = True
    fixed_params: Tuple[Tuple[str, Any], ...] = (("n_estimators", 500),)
    forest_n_jobs: int = -1

    def __post_init__(self) -> None:
        fixed = self.fixed_params
        if isinstance(fixed, Mapping):
            fixed = tuple(sorted(fixed.items()))
            object.__setattr__(self, "fixed_params", fixed)
        errors = []
        if self.n_trials < 1:
            errors.append("n_trials must be >= 1")
        if self.n_folds < 2:
            errors.append("n_folds must be >= 2")
        unknown = {k for k, _ in fixed} - set(REQUIRED_PARAM_KEYS)
        if unknown:
            errors.append(f"fixed_params has unknown keys: {sorted(unknown)}")
        if errors:
            raise ValueError("TuningBudget: " + "; ".join(errors))

    @property
    def fixed(self) -> dict:
        return dict(self.fixed_params)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fixed_params"] = self.fixed
        return d

    def signature(self) -> str:
        d = self.to_dict()
        d.pop("forest_n_jobs")
        blob = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class RouterFolds:
    """The OOF fold plan ``RFRouter.fit`` uses."""

    n_folds: int
    random_state: int
    stratify: bool = True

    @classmethod
    def from_config(cls, config) -> "RouterFolds":
        return cls(int(config.n_oof_folds), int(config.random_state),
                   bool(config.stratify_oof))
