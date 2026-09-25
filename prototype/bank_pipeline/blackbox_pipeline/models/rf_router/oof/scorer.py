"""
``OOFScorer`` — out-of-fold probabilities for a population or sub-population.

The four invariants this file exists to hold:

1. every scored row receives exactly one prediction;
2. that prediction comes from a model fitted on a fold that excluded the row;
3. the returned array is in the SAME row order as the input, positionally;
4. no fitted estimator is reused across folds.

(1), (3) and (4) are checked by ``OOFResult.assert_valid``; (2) is guaranteed by
construction below and asserted per fold.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..forest import ForestTrainer
from .folds import check_aligned, make_folds
from .results import OOFResult

__all__ = ["OOFScorer"]


class OOFScorer:
    """
    Parameters
    ----------
    trainer
        ``ForestTrainer`` supplying hyperparameters. A fresh pipeline is built
        per fold; the trainer itself holds no fitted state.
    n_folds, random_state, stratify
        Fold configuration.
    verbose
        Print per-fold progress.
    """

    def __init__(
        self,
        trainer: ForestTrainer,
        n_folds: int = 10,
        random_state: int = 42,
        stratify: bool = True,
        verbose: bool = False,
    ):
        self.trainer = trainer
        self.n_folds = int(n_folds)
        self.random_state = int(random_state)
        self.stratify = bool(stratify)
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        subset_mask: Optional[np.ndarray] = None,
        label: str = "OOF",
    ) -> OOFResult:
        """
        Out-of-fold ``P(y = 1 | x)``.

        Parameters
        ----------
        X, y
            The FULL training population. Index alignment between the two is
            checked; positions are what everything downstream uses.
        subset_mask
            Optional boolean over the full population. When given, folds are
            built WITHIN the subset and only subset rows are fitted or scored.
            This is how the Pass 2 forest is trained on the Pass 1 remainder:
            the model for a fold sees remainder rows outside that fold and
            nothing else, so no scored row contributed to its own prediction.
        """
        X, y = check_aligned(X, y)
        n_total = len(X)
        y_arr = np.asarray(y)

        mask = self._resolve_mask(subset_mask, n_total)
        sub_pos = np.flatnonzero(mask)
        if len(sub_pos) == 0:
            raise ValueError(f"{label}: subset_mask selects zero rows")

        y_sub = pd.Series(y_arr[sub_pos])
        if y_sub.nunique() < 2:
            raise ValueError(
                f"{label}: subset contains a single class "
                f"({sorted(y_sub.unique().tolist())}); cannot fit folds"
            )

        n_folds = self._resolve_n_folds(y_sub, label)
        folds = make_folds(
            y_sub, n_splits=n_folds, random_state=self.random_state,
            stratify=self.stratify,
        )

        proba = np.full(n_total, np.nan, dtype=float)
        fold_id = np.full(n_total, -1, dtype=int)
        fit_sizes: list[int] = []

        if self.verbose:
            print(f"   {label}: {len(sub_pos):,} rows, {n_folds} folds")

        for k, (tr_local, va_local) in enumerate(folds):
            # Local (within-subset) positions -> global population positions.
            tr_global = sub_pos[tr_local]
            va_global = sub_pos[va_local]

            # Construction-time guarantee for invariant (2).
            if np.intersect1d(tr_global, va_global).size != 0:
                raise AssertionError(
                    f"{label} fold {k}: train and validation positions overlap"
                )

            pipe = self.trainer.fit(
                X.iloc[tr_global],
                pd.Series(y_arr[tr_global]),
            )
            proba[va_global] = ForestTrainer.positive_proba(
                pipe, X.iloc[va_global]
            )
            fold_id[va_global] = k
            fit_sizes.append(len(tr_global))

            if self.verbose:
                print(
                    f"      fold {k + 1}/{n_folds}: "
                    f"fit {len(tr_global):,} → scored {len(va_global):,}"
                )

            del pipe  # never reused across folds

        result = OOFResult(
            proba=proba,
            scored_mask=mask,
            fold_id=fold_id,
            n_folds=n_folds,
            random_state=self.random_state,
            stratified=self.stratify,
            fit_sizes=fit_sizes,
        )
        result.assert_valid()
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_mask(subset_mask, n_total: int) -> np.ndarray:
        if subset_mask is None:
            return np.ones(n_total, dtype=bool)
        mask = np.asarray(subset_mask, dtype=bool)
        if len(mask) != n_total:
            raise ValueError(
                f"subset_mask length {len(mask)} != population size {n_total}"
            )
        return mask

    def _resolve_n_folds(self, y_sub: pd.Series, label: str) -> int:
        """Shrink the fold count when the minority class cannot fill it."""
        min_class = int(y_sub.value_counts().min())
        if self.stratify and min_class < self.n_folds:
            n_folds = max(2, min_class)
            if self.verbose:
                print(
                    f"   ⚠️  {label}: only {min_class} samples in the minority "
                    f"class — reducing folds {self.n_folds} → {n_folds}"
                )
            return n_folds
        return self.n_folds