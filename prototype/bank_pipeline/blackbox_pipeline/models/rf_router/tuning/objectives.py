"""
blackbox_pipeline.models.rf_router.tuning.objectives
=====================================================
What a tuning trial measures, and which number each objective maximises.

The specialized objectives are the router's own ``ThresholdSolver`` applied to a
configuration's OOF scores under the shared ``OperatingConstraints``. Tuning
therefore optimises exactly the decision rule that gets deployed — not a proxy
evaluated at a 0.5 cut the solver would discard.

``p1_coverage``
    Pass 1 band fraction at the constraints (largest ``t1`` meeting the NPV floor
    and both leakage caps). When the leakage cap binds, maximising coverage is
    the same as maximising NPV at the cap (NPV = 1 − L/k), so this is the
    constraint-aware "precision-oriented" Pass 1 objective too. 0 if infeasible.
``p2_recall``
    Pass 2 recall at the precision floor (smallest ``t2`` meeting the floor),
    under the constraints' recall denominator — ``global`` for GLASS
    comparability. Within one tuning run the remainder is frozen, so global and
    remainder recall rank configurations identically. 0 if infeasible.
``auc``
    Mean per-fold ROC-AUC — the baseline objective (former Cell 22), on whichever
    population the pass runs on. On the remainder it is the population control.

Every trial records ALL diagnostics regardless of which objective it optimises,
so every arm can be read on every axis afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..constraints import OperatingConstraints
from ..forest import ForestTrainer
from ..oof import OOFScorer
from ..oof.folds import check_aligned
from ..reporting import binding_after
from ..threshhold import ThresholdSolver

__all__ = ["Objective", "OBJECTIVES", "evaluate_params"]


@dataclass(frozen=True)
class Objective:
    name: str
    passes: Tuple[str, ...]
    key: str
    description: str


OBJECTIVES = {
    "auc": Objective(
        "auc", ("pass1", "pass2"), "auc_fold_mean",
        "mean per-fold ROC-AUC on the pass's population (baseline objective)"),
    "p1_coverage": Objective(
        "p1_coverage", ("pass1",), "p1_coverage",
        "Pass 1 coverage at the shared constraints; 0 if infeasible"),
    "p2_recall": Objective(
        "p2_recall", ("pass2",), "p2_recall",
        "Pass 2 recall at the precision floor; 0 if infeasible"),
}


def _fold_mean_auc(y: np.ndarray, p: np.ndarray, fold_id: np.ndarray) -> float:
    scores = [
        roc_auc_score(y[fold_id == k], p[fold_id == k])
        for k in np.unique(fold_id)
        if len(np.unique(y[fold_id == k])) == 2
    ]
    return float(np.mean(scores)) if scores else float("nan")


def _leakage_pauc(y: np.ndarray, p: np.ndarray, cap: float) -> float:
    """
    Standardised partial AUC of the Pass 1 ROUTING ranking up to the leakage cap.

    Flipping labels (non-subscriber = positive) and negating the score turns the
    false-positive axis into the leakage rate, so ``max_fpr=cap`` integrates the
    region from zero leakage to the cap.
    """
    if not 0.0 < cap:
        return float("nan")
    if cap >= 1.0:
        return float(roc_auc_score(1 - y, -p))
    return float(roc_auc_score(1 - y, -p, max_fpr=cap))


def evaluate_params(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    pass_name: str,
    constraints: OperatingConstraints,
    n_folds: int,
    fold_seed: int,
    forest_seed: int,
    stratify: bool = True,
    population_mask: Optional[np.ndarray] = None,
    n_positives_global: Optional[int] = None,
    forest_n_jobs: int = -1,
) -> dict:
    """
    Score one configuration for one pass on training OOF. Never sees test data.

    Parameters
    ----------
    params
        Normalised RF hyperparameters.
    X, y
        The FULL training population, index-aligned.
    pass_name
        ``"pass1"`` (population = all rows) or ``"pass2"`` (population =
        ``population_mask``, the Pass 1 remainder — required).
    constraints
        The shared Stage 2 operating constraints.
    n_folds, fold_seed, forest_seed, stratify
        OOF protocol for this trial (see ``budget.TuningBudget``).
    n_positives_global
        Positives in the full training population, for Pass 2's global recall
        denominator. Defaults to ``y.sum()``.

    Returns
    -------
    dict of plain Python scalars
        Shared: ``n_rows``, ``n_positives``, ``base_rate``, ``auc_fold_mean``,
        ``auc_pooled``, ``average_precision``. Pass 1: ``p1_feasible``,
        ``p1_coverage``, ``p1_npv``, ``p1_leakage_rate``,
        ``p1_subscribers_routed``, ``p1_binding``, ``p1_pauc_leak``. Pass 2:
        ``p2_feasible``, ``p2_recall``, ``p2_precision``, ``p2_band_fraction``,
        ``p2_whole_population``, ``p2_binding``.
    """
    if pass_name not in ("pass1", "pass2"):
        raise ValueError(f"pass_name must be 'pass1' or 'pass2', got {pass_name!r}")
    if pass_name == "pass2" and population_mask is None:
        raise ValueError("Pass 2 is scored on the Pass 1 remainder: pass population_mask")
    if pass_name == "pass1" and population_mask is not None:
        raise ValueError("Pass 1 runs on the full training population; no mask")

    X, y = check_aligned(X, y)
    y_all = np.asarray(y).astype(int)

    trainer = ForestTrainer(params, forest_seed, n_jobs=forest_n_jobs)
    oof = OOFScorer(trainer, n_folds, fold_seed, stratify, verbose=False).score(
        X, y, subset_mask=population_mask, label=f"{pass_name} trial OOF"
    )
    p = oof.scored_proba()
    yy = oof.scored_labels(y).astype(int)
    fid = oof.fold_id[oof.scored_mask]

    out = {
        "n_rows": int(len(p)),
        "n_positives": int(yy.sum()),
        "base_rate": float(yy.mean()),
        "auc_fold_mean": _fold_mean_auc(yy, p, fid),
        "auc_pooled": float(roc_auc_score(yy, p)),
        "average_precision": float(average_precision_score(yy, p)),
    }
    solver = ThresholdSolver(constraints)

    if pass_name == "pass1":
        r = solver.solve_pass1(p, yy)
        out.update({
            "p1_feasible": bool(r.feasible),
            "p1_coverage": float(r.band_fraction) if r.feasible else 0.0,
            "p1_npv": float(r.band_precision) if r.feasible else float("nan"),
            "p1_leakage_rate": float(r.leakage_rate) if r.feasible else float("nan"),
            "p1_subscribers_routed": int(r.positives_in_band),
            "p1_binding": binding_after(r.sweep, r.band_size) if r.feasible else "infeasible",
            "p1_pauc_leak": _leakage_pauc(yy, p, constraints.max_subscriber_leakage_rate),
        })
    else:
        npos = int(y_all.sum()) if n_positives_global is None else int(n_positives_global)
        r = solver.solve_pass2(p, yy, n_positives_global=npos)
        out.update({
            "p2_feasible": bool(r.feasible),
            "p2_recall": float(r.recall) if r.feasible else 0.0,
            "p2_precision": float(r.band_precision) if r.feasible else float("nan"),
            "p2_band_fraction": float(r.band_fraction) if r.feasible else 0.0,
            "p2_whole_population": bool(r.feasible and r.band_size == len(p)),
            "p2_binding": binding_after(r.sweep, r.band_size) if r.feasible else "infeasible",
        })
    return out
