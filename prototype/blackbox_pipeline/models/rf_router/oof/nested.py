"""
Fully nested cascade OOF — the strict check on the simple protocol.

``RFRouter.fit`` has one residual dependence: the remainder is defined by a
global ``t1`` solved using every training label. Each individual Pass 1
probability is honestly out-of-fold, but the *cut point* is not. This routine
removes even that, by re-deriving the threshold inside every outer fold.

If the operating point and headline metrics here agree with the simple
protocol, the residual dependence is immaterial and the cheap path stands. If
they diverge, report the nested numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..forest import ForestTrainer
from .folds import check_aligned, make_folds
from .scorer import OOFScorer

__all__ = ["NestedCascadeResult", "nested_cascade_oof"]


@dataclass
class NestedCascadeResult:
    """
    ``p1``/``p2`` are nested-OOF probabilities: for each outer fold, the Pass 1
    model AND the threshold defining the remainder AND the Pass 2 model were all
    derived only from rows outside that fold.
    """

    p1: np.ndarray
    p2: np.ndarray
    routed_mask: np.ndarray
    remainder_mask: np.ndarray
    fold_t1: list[float]
    fold_t2: list[float]
    fold_id: np.ndarray


def nested_cascade_oof(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    pass1_trainer: ForestTrainer,
    pass2_trainer: ForestTrainer,
    solver,
    n_outer_folds: int = 5,
    n_inner_folds: int = 5,
    random_state: int = 42,
    stratify: bool = True,
    remainder_min_size: int = 100,
    verbose: bool = True,
) -> NestedCascadeResult:
    """
    For each outer fold ``f``:

    1. run an INNER OOF on the outer-train rows and solve ``t1_f`` from it;
    2. define the outer-train remainder with ``t1_f``;
    3. solve ``t2_f`` from the inner OOF restricted to that remainder;
    4. fit Pass 1 and Pass 2 on the outer-train rows;
    5. score the held-out fold ``f`` with those models and thresholds.

    Nothing derived from a scored row — not its label, not its contribution to a
    threshold — reaches its own prediction.

    ``solver`` is a ``ThresholdSolver``; passed in rather than imported to keep
    this module free of a circular dependency.
    """
    X, y = check_aligned(X, y)
    y_arr = np.asarray(y)
    n = len(X)

    outer = make_folds(y, n_outer_folds, random_state, stratify)
    inner_seed = random_state + 1000

    def inner_scorer(trainer) -> OOFScorer:
        return OOFScorer(
            trainer, n_folds=n_inner_folds, random_state=inner_seed,
            stratify=stratify, verbose=False,
        )

    p1 = np.full(n, np.nan, dtype=float)
    p2 = np.full(n, np.nan, dtype=float)
    fold_id = np.full(n, -1, dtype=int)
    fold_t1: list[float] = []
    fold_t2: list[float] = []

    for k, (tr, va) in enumerate(outer):
        if verbose:
            print(f"\n   nested fold {k + 1}/{len(outer)}: "
                  f"fit {len(tr):,} → score {len(va):,}")

        X_tr = X.iloc[tr]
        # Index must follow X_tr, not restart at 0 — check_aligned enforces it.
        y_tr = pd.Series(y_arr[tr], index=X_tr.index)

        # --- inner OOF on the outer-train rows, for t1_f -------------------
        inner_p1 = inner_scorer(pass1_trainer).score(
            X_tr, y_tr, label=f"inner-p1[{k}]"
        ).proba

        t1_res = solver.solve_pass1(inner_p1, y_tr.to_numpy())
        t1_res.raise_if_infeasible(context=f"nested fold {k} Pass 1")
        t1_f = t1_res.threshold
        fold_t1.append(float(t1_f))

        # --- remainder of the outer-train rows, from INNER OOF decisions ---
        rem_local = inner_p1 >= t1_f
        if int(rem_local.sum()) < remainder_min_size:
            raise ValueError(
                f"nested fold {k}: remainder has {int(rem_local.sum())} rows, "
                f"below remainder_min_size={remainder_min_size}"
            )

        inner_p2 = inner_scorer(pass2_trainer).score(
            X_tr, y_tr, subset_mask=rem_local, label=f"inner-p2[{k}]"
        )
        t2_res = solver.solve_pass2(
            inner_p2.scored_proba(),
            inner_p2.scored_labels(y_tr),
            n_positives_global=int((y_tr == 1).sum()),
        )
        t2_res.raise_if_infeasible(context=f"nested fold {k} Pass 2")
        fold_t2.append(float(t2_res.threshold))

        # --- outer models, fitted only on outer-train ----------------------
        rem_idx = np.flatnonzero(rem_local)
        m1 = pass1_trainer.fit(X_tr, y_tr)
        m2 = pass2_trainer.fit(
            X_tr.iloc[rem_idx], pd.Series(y_arr[tr][rem_local])
        )

        # --- score the held-out fold ---------------------------------------
        X_va = X.iloc[va]
        p1_va = ForestTrainer.positive_proba(m1, X_va)
        p1[va] = p1_va
        fold_id[va] = k

        rem_va = p1_va >= t1_f
        if rem_va.any():
            p2[va[rem_va]] = ForestTrainer.positive_proba(
                m2, X_va.iloc[np.flatnonzero(rem_va)]
            )

        del m1, m2

    # Routing uses per-fold thresholds: each fold has its own t1, so a single
    # global cut would be wrong here.
    routed = np.zeros(n, dtype=bool)
    for k, (_, va) in enumerate(outer):
        routed[va] = p1[va] < fold_t1[k]

    return NestedCascadeResult(
        p1=p1,
        p2=p2,
        routed_mask=routed,
        remainder_mask=~routed,
        fold_t1=fold_t1,
        fold_t2=fold_t2,
        fold_id=fold_id,
    )