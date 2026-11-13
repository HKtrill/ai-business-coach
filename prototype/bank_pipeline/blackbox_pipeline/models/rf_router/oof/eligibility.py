"""
``nested_pass1_eligibility`` — Pass 2 eligibility with no row's own label in its
cut, at router level (conditional on the engineered features; see below).

PR 37 Commit 3 (F17). Opt-in; the historical router does not call this.

The historical rule (``RFRouter.fit``, ``eligibility="global_t1"``)
-----------------------------------------------------------------
Every training row gets an out-of-fold Pass 1 probability, from a forest that
never saw it. One cut ``t1`` is then solved from ALL those probabilities and
ALL their labels, and the Pass 2 training remainder is ``p1_oof >= t1``. So row
*i*'s label takes part in choosing the cut that decides whether row *i* joins
the Pass 2 training population: a selection dependence through one scalar.

The nested rule (``eligibility="nested_t1"``)
---------------------------------------------
Reuse the inner fold plan the Pass 1 OOF already used. For inner fold *j*:

1. the rows OUTSIDE *j* are scored out of fold among themselves, by forests
   fitted only on rows outside *j* (``OOFScorer`` with ``subset_mask``);
2. ``t1_j`` is solved from those scores and labels only;
3. the rows IN *j* are eligible when their existing Pass 1 OOF probability
   (from the forest fitted on the rows outside *j*) is ``>= t1_j``.

Row *i* in fold *j* is therefore decided by a forest fitted without it and a
cut solved without it: no router component reads ``y_i``. Rows outside *j*
never use fold *j*'s cut, so no fold's labels decide their own fold's
eligibility through the router.

Scope of the guarantee (PR 37 F17, limitation L1)
-------------------------------------------------
The isolation is router-level and conditional on the engineered features. The
cross-fits engineer features once per outer fold, on the whole outer-training
partition and its labels (``shared.stage2.feature_engineering.engineer_features``;
``neighborhood_subscription_density`` is a KDE over training labels). A row's
own label can therefore reach its own feature values, and through them its OOF
scores and the scores ``t1_j`` is solved on. Mandatory PR 38 gate G13 audits
this; ``documentation/methodology/pr37_f17_decision_record.md`` records the
limitation.

What it does not change
-----------------------
The deployed ``t1`` (applied to rows the router has never seen) is still the
one solved on every OOF probability; its rows are outside the fitting
population, so no selection dependence arises there. ``t2`` and the band
confidences are solved on the resulting remainder as before.

Cost
----
``n_folds`` extra Pass 1 forest fits for each inner fold: with 5 inner folds
and ``n_folds = 5`` that is 25 extra Pass 1 fits per router fit (on about 80%
of the population each, each fold model on about 64%), on top of the 5 OOF
fits and the refit.

Score scale: ``t1_j`` is solved on scores from forests fitted on about 64% of
the population and applied to scores from a forest fitted on about 80%. The
same nested-CV mismatch is accepted by ``oof.nested_cascade_oof``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import ELIGIBILITY_GLOBAL_T1, ELIGIBILITY_MODES, ELIGIBILITY_NESTED_T1
from .folds import check_aligned
from .scorer import OOFScorer

__all__ = [
    "ELIGIBILITY_GLOBAL_T1",
    "ELIGIBILITY_NESTED_T1",
    "ELIGIBILITY_MODES",
    "NESTED_SOURCE",
    "NestedEligibility",
    "nested_pass1_eligibility",
]

#: ``fit_report_.remainder["source"]`` under the nested rule. Router-level
#: wording: the features were engineered on the outer-training partition (L1).
NESTED_SOURCE = ("OOF pass1 probabilities >= fold-nested t1 (router-level: no row's own "
                 "label in its cut; features engineered on the outer-training partition)")


@dataclass(frozen=True)
class NestedEligibility:
    """
    remainder
        Boolean over the population: rows eligible for Pass 2 training.
    fold_t1
        The cut solved for each inner fold, in fold order.
    fold_rows / fold_remainder
        Rows in each inner fold, and how many of them are eligible.
    n_folds, random_state, stratified
        The within-subset fold plan used to solve each ``t1_j``.
    fold_ids
        The inner fold each entry above belongs to (``fold_id`` values, sorted).
    fold_counts
        PR 38 (G5): per inner fold, the population ``t1_j`` was solved on (the
        rows outside fold j) and its band ``p < t1_j``, recounted from the
        scores independently of the solver: ``population_rows``,
        ``population_positives``, ``band_rows``, ``band_positives``.
    """

    remainder: np.ndarray
    fold_t1: Tuple[float, ...]
    fold_rows: Tuple[int, ...]
    fold_remainder: Tuple[int, ...]
    n_folds: int
    random_state: int
    stratified: bool
    fold_ids: Tuple[int, ...] = ()
    fold_counts: Tuple[dict, ...] = ()

    def to_report(self) -> dict:
        return {
            "eligibility_fold_t1": [float(t) for t in self.fold_t1],
            "eligibility_fold_rows": [int(v) for v in self.fold_rows],
            "eligibility_fold_remainder": [int(v) for v in self.fold_remainder],
            "eligibility_folds": int(self.n_folds),
            "eligibility_random_state": int(self.random_state),
            "eligibility_stratified": bool(self.stratified),
            "eligibility_fold_ids": [int(j) for j in self.fold_ids],
            "eligibility_fold_counts": [dict(c) for c in self.fold_counts],
        }


def nested_pass1_eligibility(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    pass1_proba: np.ndarray,
    fold_id: np.ndarray,
    trainer,
    solver,
    n_folds: int,
    random_state: int,
    stratify: bool = True,
    verbose: bool = False,
    fold_features: Optional[Callable[[np.ndarray], Mapping[int, pd.DataFrame]]] = None,
) -> NestedEligibility:
    """
    Pass 2 eligibility in which no row's own label reaches its own cut through
    the router (conditional on the engineered features; module docstring, L1).

    Parameters
    ----------
    X, y
        The population Pass 1 was scored on (index-aligned).
    pass1_proba
        Its Pass 1 OOF probabilities (``OOFResult.proba``), in population order.
    fold_id
        The inner fold that held out each row in that OOF run
        (``OOFResult.fold_id``). Every row must belong to exactly one fold.
    trainer
        The Pass 1 ``ForestTrainer``.
    solver
        A ``ThresholdSolver`` holding the router's constraints.
    n_folds, random_state, stratify
        Fold plan for scoring the rows outside each inner fold.
    fold_features
        PR 38 (G13), opt-in: ``fold_features(fold_id)[j]`` are inner fold j's
        features for every row, computed without fold j's labels. ``t1_j`` is
        then solved from forests fitted and scored on them. None: ``X``.

    Raises
    ------
    ValueError
        Misaligned inputs, a fold plan that does not partition the rows, or an
        infeasible ``t1_j``. There is no fallback to the global cut.
    """
    X, y = check_aligned(X, y)
    y_arr = np.asarray(y).astype(int)
    n = len(y_arr)
    p1 = np.asarray(pass1_proba, dtype=float)
    fold = np.asarray(fold_id)

    if p1.shape != (n,) or fold.shape != (n,):
        raise ValueError(
            f"pass1_proba {p1.shape} and fold_id {fold.shape} must both have shape ({n},)"
        )
    if not np.isfinite(p1).all():
        raise ValueError("pass1_proba must be finite for every row (score the full population)")
    folds = np.unique(fold)
    if (fold < 0).any() or len(folds) < 2:
        raise ValueError("fold_id must assign every row to one of at least two inner folds")

    scorer = OOFScorer(trainer, int(n_folds), int(random_state), bool(stratify), verbose=False)
    remainder = np.zeros(n, dtype=bool)
    fold_t1, fold_rows, fold_rem, fold_counts = [], [], [], []

    for j in folds:
        in_j = fold == j
        outside = ~in_j
        # Scores for rows outside j, from forests fitted only on rows outside j.
        X_j = X if fold_features is None else fold_features(fold)[int(j)]
        sub = scorer.score(X_j, y, subset_mask=outside, label=f"eligibility[{int(j)}]")
        res = solver.solve_pass1(sub.scored_proba(), sub.scored_labels(y), keep_sweep=False)
        res.raise_if_infeasible(context=f"nested eligibility, inner fold {int(j)}")
        t1_j = float(res.threshold)
        # PR 38 (G5): the band t1_j selects, recounted from the scores. The
        # solver's own counts must agree, or its feasibility claim is unverified.
        sub_p = np.asarray(sub.scored_proba(), dtype=float)
        sub_y = np.asarray(sub.scored_labels(y)).astype(int)
        band = sub_p < t1_j
        counts = {"population_rows": int(len(sub_y)), "population_positives": int(sub_y.sum()),
                  "band_rows": int(band.sum()), "band_positives": int(sub_y[band].sum())}
        if (counts["band_rows"], counts["band_positives"]) != (int(res.band_size),
                                                               int(res.positives_in_band)):
            raise AssertionError(
                f"nested eligibility, inner fold {int(j)}: recount {counts} disagrees with the "
                f"solver (band {res.band_size}, positives {res.positives_in_band})")
        fold_counts.append(counts)
        remainder[in_j] = p1[in_j] >= t1_j
        fold_t1.append(t1_j)
        fold_rows.append(int(in_j.sum()))
        fold_rem.append(int(remainder[in_j].sum()))
        if verbose:
            print(f"   eligibility fold {int(j)}: t1_j {t1_j:.6f} | "
                  f"{fold_rem[-1]:,}/{fold_rows[-1]:,} rows eligible")

    return NestedEligibility(
        remainder=remainder,
        fold_t1=tuple(fold_t1),
        fold_rows=tuple(fold_rows),
        fold_remainder=tuple(fold_rem),
        n_folds=int(n_folds),
        random_state=int(random_state),
        stratified=bool(stratify),
        fold_ids=tuple(int(j) for j in folds),
        fold_counts=tuple(fold_counts),
    )
