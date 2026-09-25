"""
shared.stage4.protocol
======================
The abstention rule Stage 4 arbiters are held to, stated once.

The GLASS Meta-EBM tunes ``min_weighted_confidence`` with
``glass_pipeline.meta_ebm.tuning.tune_arbiter_threshold``:

    grid        np.arange(0.03, 0.55, 0.02)
    objective   F2 on the retained (non-abstained) rows
    constraint  retained coverage ≥ 0.50
    selection   strictly-greater F2 wins (ties → the earlier, smaller value);
                nothing qualifies → None
    data        training rows only (Stage 4 train-side inputs)

``tune_min_confidence`` applies the same rule to any model that emits a
confidence per row, e.g. a learned meta-model's |p − t|. A test pins these
constants to the Meta-EBM helper's defaults, so the two arms cannot drift.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import fbeta_score

ABSTENTION_GRID = np.arange(0.03, 0.55, 0.02)
ABSTENTION_MIN_COVERAGE = 0.50


def tune_min_confidence(
    y_true: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray,
    grid: Optional[np.ndarray] = None,
    min_coverage: float = ABSTENTION_MIN_COVERAGE,
) -> Optional[dict]:
    """
    Best ``min_confidence`` by F2 on retained rows, subject to coverage.

    Abstain where ``confidence < min_confidence`` (the Meta-EBM rule:
    ``max(conf1, conf0) < min_weighted_confidence``). Returns
    {min_confidence, train_f2, train_coverage, sweep} or None.
    """
    y = np.asarray(y_true).astype(int)
    d = np.asarray(pred).astype(int)
    c = np.asarray(confidence, dtype=float)
    if not (len(y) == len(d) == len(c)):
        raise ValueError("y / pred / confidence lengths differ")
    grid = ABSTENTION_GRID if grid is None else np.asarray(grid, dtype=float)

    best_score, best, sweep = 0.0, None, []
    for mc in grid:
        covered = c >= mc
        cov = float(covered.mean())
        f2 = float(fbeta_score(y[covered], d[covered], beta=2, zero_division=0)) \
            if covered.any() else 0.0
        sweep.append({"min_confidence": float(mc), "coverage": cov, "f2": f2,
                      "eligible": cov >= min_coverage})
        if cov < min_coverage:
            continue
        if f2 > best_score:
            best_score = f2
            best = {"min_confidence": float(mc), "train_f2": f2, "train_coverage": cov}
    if best is not None:
        best["sweep"] = sweep
    return best
