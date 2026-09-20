"""
blackbox_pipeline.models.rf_router.oof
=======================================
Leak-free out-of-fold probability generation.

Every threshold in this router is selected from the arrays this module
produces. That makes it the single most leakage-sensitive file in the package,
so the invariants are asserted here rather than inspected later:

1. every scored row receives exactly one prediction;
2. that prediction comes from a model fitted on a fold that excluded the row;
3. the returned array is in the SAME row order as the input, positionally;
4. no fitted estimator is reused across folds.

Why OOF at all
--------------
``train_rf_stage`` refits on the full ``X_train``, and in-sample
``predict_proba`` on a forest of that depth is close to separating. Thresholds
read off those numbers would place ``t1`` and ``t2`` almost on top of each
other, produce a near-zero abstain band on train, and collapse on test. OOF
probabilities are what the model would have said about a row it had not seen,
which is the right basis for choosing an operating point.

This is not an advantage over GLASS. GLASS selects its rules against
``X_val``/``y_val``, which ``GLASSBRWPipeline.fit`` defaults to the training
data when no validation split is passed. Using OOF here gives the RF a *less*
optimistic view of its own training data than GLASS grants itself. If the GLASS
arm is later re-run with a genuine validation split, revisit this note; the
comparison stays honest either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .forest import ForestTrainer

__all__ = [
    "OOFResult",
    "OOFScorer",
    "make_folds",
    "nested_cascade_oof",
    "NestedCascadeResult",
]


# ======================================================================
# Fold construction
# ======================================================================

def make_folds(
    y: pd.Series,
    n_splits: int,
    random_state: int,
    stratify: bool = True,
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """
    Positional (train_idx, val_idx) pairs over ``range(len(y))``.

    Indices are POSITIONAL, never pandas labels. Mixing the two is the
    fold-misalignment bug this package is built to avoid, so every consumer
    slices with ``.iloc`` / numpy fancy indexing only.
    """
    y_arr = np.asarray(y)
    n = len(y_arr)
    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if stratify
        else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )
    dummy = np.zeros((n, 1))
    return [
        (np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
        for tr, va in splitter.split(dummy, y_arr)
    ]


# ======================================================================
# Result container
# ======================================================================

@dataclass
class OOFResult:
    """
    Out-of-fold probabilities over a scored population.

    Attributes
    ----------
    proba
        ``P(y = 1 | x)``, length ``n_rows`` of the FULL population handed to the
        scorer. Positions outside ``scored_mask`` hold ``np.nan``.
    scored_mask
        Boolean over the full population: which rows were scored. For a
        full-population run this is all True; for a remainder run it is the
        remainder.
    fold_id
        Which fold held out each scored row. ``-1`` where unscored. Used by the
        leakage assertions and by the nested diagnostic.
    n_folds, random_state, stratified
        Provenance of the fold assignment.
    """

    proba: np.ndarray
    scored_mask: np.ndarray
    fold_id: np.ndarray
    n_folds: int
    random_state: int
    stratified: bool
    fit_sizes: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def n_scored(self) -> int:
        return int(self.scored_mask.sum())

    def scored_proba(self) -> np.ndarray:
        """Probabilities for scored rows only, in population order."""
        return self.proba[self.scored_mask]

    def scored_labels(self, y: pd.Series) -> np.ndarray:
        """Labels for scored rows only, in population order."""
        return np.asarray(y)[self.scored_mask]

    # ------------------------------------------------------------------
    def assert_valid(self) -> None:
        """
        Structural invariants. Cheap; called on every construction.

        Does not and cannot prove that a fold model never saw a row — that is
        enforced by construction in ``OOFScorer`` and verified independently by
        ``tests/test_rf_router_leakage.py``.
        """
        n = len(self.proba)
        if not (len(self.scored_mask) == len(self.fold_id) == n):
            raise AssertionError(
                "OOFResult array length mismatch: "
                f"proba={n}, mask={len(self.scored_mask)}, fold={len(self.fold_id)}"
            )

        scored = self.scored_mask
        if np.isnan(self.proba[scored]).any():
            raise AssertionError(
                "OOF probability is NaN for a row marked as scored — a fold "
                "failed to write its predictions."
            )
        if not np.isnan(self.proba[~scored]).all():
            raise AssertionError(
                "OOF probability present for a row marked as unscored — the "
                "scored mask and the probability array disagree."
            )
        if (self.fold_id[scored] < 0).any():
            raise AssertionError("Scored row has no fold assignment.")
        if (self.fold_id[~scored] != -1).any():
            raise AssertionError("Unscored row carries a fold assignment.")

        bad = (self.proba[scored] < 0.0) | (self.proba[scored] > 1.0)
        if bad.any():
            raise AssertionError(
                f"{int(bad.sum())} OOF probabilities outside [0, 1]."
            )

        counts = np.bincount(
            self.fold_id[scored], minlength=self.n_folds
        )
        if len(counts) != self.n_folds or (counts == 0).any():
            raise AssertionError(
                f"Expected {self.n_folds} non-empty folds, got counts {counts.tolist()}"
            )
        if int(counts.sum()) != int(scored.sum()):
            raise AssertionError(
                "Fold assignments do not partition the scored rows: "
                f"{int(counts.sum())} assignments for {int(scored.sum())} rows."
            )


# ======================================================================
# Scorer
# ======================================================================

class OOFScorer:
    """
    Produces out-of-fold probabilities for a population or a sub-population.

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
        X, y = _check_aligned(X, y)
        n_total = len(X)

        if subset_mask is None:
            mask = np.ones(n_total, dtype=bool)
        else:
            mask = np.asarray(subset_mask, dtype=bool)
            if len(mask) != n_total:
                raise ValueError(
                    f"subset_mask length {len(mask)} != population size {n_total}"
                )

        sub_pos = np.flatnonzero(mask)
        if len(sub_pos) == 0:
            raise ValueError(f"{label}: subset_mask selects zero rows")

        X_sub = X.iloc[sub_pos]
        y_sub = pd.Series(np.asarray(y)[sub_pos])

        if y_sub.nunique() < 2:
            raise ValueError(
                f"{label}: subset contains a single class "
                f"({sorted(y_sub.unique().tolist())}); cannot fit folds"
            )

        min_class = int(y_sub.value_counts().min())
        n_folds = self.n_folds
        if self.stratify and min_class < n_folds:
            n_folds = max(2, min_class)
            if self.verbose:
                print(
                    f"   ⚠️  {label}: only {min_class} samples in the minority "
                    f"class — reducing folds {self.n_folds} → {n_folds}"
                )

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
                pd.Series(np.asarray(y)[tr_global]),
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


# ======================================================================
# Nested cascade diagnostic
# ======================================================================

@dataclass
class NestedCascadeResult:
    """
    Output of the fully nested two-pass OOF diagnostic.

    ``p1``/``p2`` are nested-OOF probabilities: for each outer fold, both the
    Pass 1 model AND the threshold that defines the remainder AND the Pass 2
    model were derived only from rows outside that fold.
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
    Fully nested cascade OOF — the strict check on the simple protocol.

    The simple protocol in ``RFRouter.fit`` has one residual dependence: the
    remainder is defined by a global ``t1`` that was solved using every training
    label. Each individual Pass 1 probability is honestly out-of-fold, but the
    *cut point* is not. This routine removes even that:

    For each outer fold ``f``:
        1. fit Pass 1 on the outer-train rows;
        2. run an INNER OOF on the outer-train rows and solve ``t1_f`` from it;
        3. define the outer-train remainder with ``t1_f``;
        4. fit Pass 2 on that remainder and solve ``t2_f`` from the inner OOF
           restricted to the remainder;
        5. score the held-out fold ``f`` with those models and thresholds.

    Nothing derived from a scored row — not its label, not its contribution to a
    threshold — reaches its own prediction. If the operating point and headline
    metrics from this procedure agree with the simple protocol, the residual
    dependence is immaterial and the cheap path stands. If they diverge, report
    the nested numbers.

    ``solver`` is a ``ThresholdSolver``; passed in rather than imported to keep
    this module free of a circular dependency.
    """
    X, y = _check_aligned(X, y)
    y_arr = np.asarray(y)
    n = len(X)

    outer = make_folds(y, n_outer_folds, random_state, stratify)

    p1 = np.full(n, np.nan, dtype=float)
    p2 = np.full(n, np.nan, dtype=float)
    fold_id = np.full(n, -1, dtype=int)
    fold_t1: list[float] = []
    fold_t2: list[float] = []

    inner_scorer_seed = random_state + 1000

    for k, (tr, va) in enumerate(outer):
        if verbose:
            print(f"\n   nested fold {k + 1}/{len(outer)}: "
                  f"fit {len(tr):,} → score {len(va):,}")

        X_tr = X.iloc[tr]
        # Index must follow X_tr, not restart at 0 — _check_aligned enforces it.
        y_tr = pd.Series(y_arr[tr], index=X_tr.index)

        # --- inner OOF on the outer-train rows, for t1_f ------------------
        inner = OOFScorer(
            pass1_trainer, n_folds=n_inner_folds,
            random_state=inner_scorer_seed, stratify=stratify, verbose=False,
        )
        inner_p1 = inner.score(X_tr, y_tr, label=f"inner-p1[{k}]").proba

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

        inner2 = OOFScorer(
            pass2_trainer, n_folds=n_inner_folds,
            random_state=inner_scorer_seed, stratify=stratify, verbose=False,
        )
        inner_p2 = inner2.score(
            X_tr, y_tr, subset_mask=rem_local, label=f"inner-p2[{k}]"
        )
        t2_res = solver.solve_pass2(
            inner_p2.scored_proba(),
            inner_p2.scored_labels(y_tr),
            n_positives_global=int((y_tr == 1).sum()),
        )
        t2_res.raise_if_infeasible(context=f"nested fold {k} Pass 2")
        t2_f = t2_res.threshold
        fold_t2.append(float(t2_f))

        # --- outer models, fitted only on outer-train ---------------------
        m1 = pass1_trainer.fit(X_tr, y_tr)
        m2 = pass2_trainer.fit(X_tr.iloc[np.flatnonzero(rem_local)],
                               pd.Series(y_arr[tr][rem_local]))

        # --- score the held-out fold --------------------------------------
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


# ======================================================================
# Helpers
# ======================================================================

def _check_aligned(X: pd.DataFrame, y) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Enforce that X and y describe the same rows in the same order.

    Length equality is not enough: a y whose index was re-sorted upstream would
    pass a length check and silently scramble every label. When both carry a
    pandas index, the indexes must be identical.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a DataFrame, got {type(X).__name__}")

    if isinstance(y, pd.Series):
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
        if not X.index.equals(y.index):
            raise ValueError(
                "X.index and y.index differ. Row order must match positionally; "
                "reindex y to X before calling (y = y.loc[X.index])."
            )
        return X, y

    y_arr = np.asarray(y)
    if len(X) != len(y_arr):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y_arr)}")
    return X, pd.Series(y_arr, index=X.index)
