"""
blackbox_pipeline.models.xgb.oof
=================================
Leakage-safe out-of-fold Stage 3 predictions.

This is the module Stage 4 depends on. Every number it produces satisfies:

1. every training row receives exactly one probability;
2. that probability comes from a model fitted on folds that excluded the row;
3. the array is in the SAME positional order as the training split;
4. no fitted booster is reused across folds.

Why this matters more here than in the EBM
------------------------------------------
GLASS Stage 3 stores ``model.predict_proba(X_train)`` from the full-data refit,
which is in-sample: the model has already seen every row it is scoring. A
Stage 4 arbiter trained on those columns learns from memorised output and will
look better in training than it can possibly be in deployment. PR 3 requires the
black-box arm to produce honest OOF columns instead, so the stage emits BOTH
blocks and labels them unambiguously — see ``artifacts.Stage3Artifact``.

The fold partition is the same ``FoldPlan`` the Optuna search used, so the
hyperparameters and the OOF predictions rest on one declared partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from .estimator import XGBFactory
from .folds import FoldPlan
from .weights import BalancedWeights

__all__ = ["OOFPredictions", "OOFGenerator"]


@dataclass
class OOFPredictions:
    """Out-of-fold probabilities over the training split."""

    proba: np.ndarray
    fold_id: np.ndarray
    n_folds: int
    random_state: int
    params: dict[str, Any]
    fit_sizes: list[int] = field(default_factory=list)
    fold_models: Optional[list] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    @property
    def n_rows(self) -> int:
        return len(self.proba)

    def assert_valid(self) -> None:
        n = self.n_rows
        if len(self.fold_id) != n:
            raise AssertionError(
                f"OOF length mismatch: proba={n}, fold_id={len(self.fold_id)}"
            )
        if np.isnan(self.proba).any():
            raise AssertionError(
                f"{int(np.isnan(self.proba).sum())} training rows have no "
                "out-of-fold prediction — a fold failed to write its output."
            )
        bad = (self.proba < 0.0) | (self.proba > 1.0)
        if bad.any():
            raise AssertionError(
                f"{int(bad.sum())} out-of-fold probabilities outside [0, 1]."
            )
        if (self.fold_id < 0).any():
            raise AssertionError("A scored row carries no fold assignment.")
        counts = np.bincount(self.fold_id, minlength=self.n_folds)
        if len(counts) != self.n_folds or (counts == 0).any():
            raise AssertionError(
                f"Expected {self.n_folds} non-empty folds, got {counts.tolist()}"
            )
        if int(counts.sum()) != n:
            raise AssertionError("Fold assignments do not partition the rows.")

    def summary(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "fit_sizes": list(self.fit_sizes),
            "proba_mean": float(self.proba.mean()),
            "proba_std": float(self.proba.std()),
            "proba_min": float(self.proba.min()),
            "proba_max": float(self.proba.max()),
        }


class OOFGenerator:
    """
    Refits the tuned configuration once per fold and scores the held-out rows.

    Parameters
    ----------
    factory  Builds a fresh booster per fold.
    weights  Balanced weights over the full training split, sliced per fold —
             the identical treatment the tuning phase applied.
    folds    The canonical partition.
    """

    def __init__(
        self,
        factory: XGBFactory,
        weights: BalancedWeights,
        folds: FoldPlan,
        verbose: bool = False,
    ):
        self.factory = factory
        self.weights = weights
        self.folds = folds
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    def generate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict[str, Any],
        keep_fold_models: bool = False,
    ) -> OOFPredictions:
        self.folds.assert_matches(len(X), "OOF generation")
        self.weights.assert_matches(len(X), "OOF generation")

        proba = np.full(len(X), np.nan, dtype=float)
        fit_sizes: list[int] = []
        models: list = []

        if self.verbose:
            print(f"\n  out-of-fold predictions: {len(X):,} rows, "
                  f"{self.folds.n_splits} folds")

        for k, (train_idx, val_idx) in enumerate(self.folds):
            # Invariant (2), guaranteed by construction and checked anyway.
            if np.intersect1d(train_idx, val_idx).size:
                raise AssertionError(
                    f"fold {k}: train and validation positions overlap"
                )

            model = self.factory.fit(
                params, X.iloc[train_idx], y.iloc[train_idx],
                sample_weight=self.weights.for_rows(train_idx),
            )
            proba[val_idx] = XGBFactory.positive_proba(model, X.iloc[val_idx])
            fit_sizes.append(int(len(train_idx)))

            if self.verbose:
                print(f"     fold {k + 1}/{self.folds.n_splits}: "
                      f"fit {len(train_idx):,} → scored {len(val_idx):,}")

            if keep_fold_models:
                models.append(model)
            else:
                del model  # never reused across folds

        result = OOFPredictions(
            proba=proba,
            fold_id=self.folds.fold_id.copy(),
            n_folds=self.folds.n_splits,
            random_state=self.folds.random_state,
            params=dict(params),
            fit_sizes=fit_sizes,
            fold_models=models if keep_fold_models else None,
        )
        result.assert_valid()
        return result