"""
blackbox_pipeline.models.rf_router.tuning.stage
================================================
The Pass 1 stage of the cascade, reproduced outside the router.

Pass 2 tuning needs the population Pass 2 will actually be trained on — the
remainder a given Pass 1 configuration leaves. ``pass1_stage`` runs exactly
``RFRouter.fit`` steps 1-3:

    1. Pass 1 OOF over the full training split (router folds, router seed)
    2. t1 solved from those OOF probabilities
    3. remainder := OOF Pass 1 probability >= t1

so the mask it returns IS ``RFRouter(...).fit(...).remainder_mask_`` for the same
Pass 1 parameters (``tests/test_rf_router_specialization.py`` pins that).

Remainder coupling: change the Pass 1 parameters and the remainder changes, so
Pass 2 parameters tuned on one remainder say nothing about another.
``Pass1Stage.remainder_fingerprint`` is stamped on every Pass 2 tuning result
and re-checked against the fitted arm router (``ablation.assert_arm_coupling``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..constraints import OperatingConstraints
from ..forest import ForestTrainer
from ..oof import OOFScorer
from ..oof.folds import check_aligned
from ..threshhold import ThresholdResult, ThresholdSolver
from .budget import RouterFolds

__all__ = ["Pass1Stage", "pass1_stage", "mask_fingerprint"]


def mask_fingerprint(mask: np.ndarray) -> str:
    """Stable 16-hex digest of a boolean row mask (positional)."""
    m = np.asarray(mask, dtype=bool)
    h = hashlib.sha256()
    h.update(str(len(m)).encode())
    h.update(np.packbits(m).tobytes())
    return h.hexdigest()[:16]


@dataclass
class Pass1Stage:
    """
    One Pass 1 configuration's operating point and the remainder it leaves.

    Attributes
    ----------
    params
        Pass 1 hyperparameters.
    p1_oof, fold_id
        Pass 1 OOF probabilities and the fold that held out each row.
    t1
        The solved Pass 1 threshold (training OOF only).
    remainder_mask
        Rows Pass 1 leaves: ``p1_oof >= t1``.
    folds
        The router fold plan that produced ``p1_oof``.
    """

    params: dict
    p1_oof: np.ndarray
    fold_id: np.ndarray
    t1: ThresholdResult
    remainder_mask: np.ndarray
    folds: RouterFolds
    source: str = field(default="pass1_stage")

    @property
    def n_remainder(self) -> int:
        return int(self.remainder_mask.sum())

    @property
    def remainder_fingerprint(self) -> str:
        return mask_fingerprint(self.remainder_mask)

    def summary(self, y) -> dict:
        y = np.asarray(y).astype(int)
        rem = self.remainder_mask
        return {
            "t1": float(self.t1.threshold),
            "pass1 coverage": float(self.t1.band_fraction),
            "remainder rows": self.n_remainder,
            "remainder fraction": float(rem.mean()),
            "remainder base rate": float(y[rem].mean()),
            "remainder fingerprint": self.remainder_fingerprint,
        }

    @classmethod
    def from_router(cls, router) -> "Pass1Stage":
        """
        The stage a FITTED ``RFRouter`` already computed — no refit.

        Use this for the current baseline (arm A0): its remainder is already on
        the router, so recomputing it would spend ``n_oof_folds`` forest fits to
        get the same mask back.
        """
        if not getattr(router, "is_fitted", False) or router.remainder_mask_ is None:
            raise ValueError("from_router needs a fitted two-pass RFRouter")
        return cls(
            params=dict(router.config.rf_params),
            p1_oof=np.asarray(router.oof_pass1_proba_),
            fold_id=np.asarray(router.oof_pass1_fold_id_),
            t1=router.thresholds_.pass1,
            remainder_mask=np.asarray(router.remainder_mask_, dtype=bool),
            folds=RouterFolds.from_config(router.config),
            source="router",
        )


def pass1_stage(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    constraints: OperatingConstraints,
    folds: RouterFolds,
    *,
    forest_n_jobs: int = -1,
    remainder_min_size: Optional[int] = None,
) -> Pass1Stage:
    """
    ``RFRouter.fit`` steps 1-3 for ``params``. Training data only.

    Raises
    ------
    ValueError
        If ``t1`` is infeasible, the remainder is single-class, or it is smaller
        than ``remainder_min_size`` — the same conditions under which the router
        itself would refuse to fit.
    """
    X, y = check_aligned(X, y)
    y_arr = np.asarray(y).astype(int)

    trainer = ForestTrainer(params, folds.random_state, n_jobs=forest_n_jobs)
    oof = OOFScorer(trainer, folds.n_folds, folds.random_state, folds.stratify,
                    verbose=False).score(X, y, label="pass1-stage OOF")
    p1 = oof.scored_proba()

    t1 = ThresholdSolver(constraints).solve_pass1(p1, y_arr)
    t1.raise_if_infeasible(context="pass1_stage")
    remainder = p1 >= t1.threshold

    n_rem = int(remainder.sum())
    rem_pos = int(y_arr[remainder].sum())
    if remainder_min_size is not None and n_rem < remainder_min_size:
        raise ValueError(f"remainder has {n_rem} rows < remainder_min_size={remainder_min_size}")
    if rem_pos == 0 or rem_pos == n_rem:
        raise ValueError(f"remainder is single-class ({rem_pos} positives of {n_rem})")

    return Pass1Stage(
        params=dict(params), p1_oof=p1, fold_id=oof.fold_id, t1=t1,
        remainder_mask=remainder, folds=folds,
    )
