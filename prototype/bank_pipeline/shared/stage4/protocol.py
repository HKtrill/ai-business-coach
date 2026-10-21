"""
shared.stage4.protocol
======================
The abstention rule Stage 4 arbiters are held to, stated once.

Both Stage 4 arms choose their abstention cut with ``tune_min_confidence``:
the GLASS Meta-EBM on its arbiter's weighted confidence
(``glass_pipeline.meta_ebm.tuning.tune_arbiter_threshold`` delegates here),
Meta-XGB on its meta-model's |p − t|.

    rule        abstain where confidence < cut (retain confidence ≥ cut)
    candidates  every distinct confidence value on the training rows
                (exact search; PR 36 F30)
    objective   F2 on the retained (non-abstained) rows
    constraint  retained coverage ≥ 0.50
    selection   highest F2 among eligible cuts; ties → the smaller cut
                (higher coverage); no eligible cut → None
    data        training rows only (Stage 4 train-side inputs)

Why exact (F30)
---------------
Coverage and retained-row F2 change only where the cut crosses an observed
confidence value, so the distinct values enumerate every retained set the
rule can produce: the search is exhaustive, has no resolution or range
parameter, and does not depend on a confidence's scale. The fixed grid
``ABSTENTION_GRID`` (0.03–0.53, step 0.02) it replaces gave each arm a
different effective budget: after PR 36 F25 the Meta-EBM's weighted
confidence has a median near 0.04, leaving one eligible grid value (0.03,
~68% coverage) instead of the constrained optimum. The grid is kept for
callers that pass it explicitly; the rule applied to it is the same.

A test pins both arms to this function, so the two arms cannot drift.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

ABSTENTION_GRID = np.arange(0.03, 0.55, 0.02)          # pre-F30 default; explicit use only
ABSTENTION_MIN_COVERAGE = 0.50
ABSTENTION_SEARCH = ("exact: every distinct training confidence value; abstain where "
                     "confidence < cut (PR 36 F30)")


def _retained_curve(y: np.ndarray, d: np.ndarray, c: np.ndarray, cuts: np.ndarray):
    """Coverage and F2 of the retained rows (c ≥ cut) for each cut, ascending cuts."""
    order = np.argsort(c, kind="stable")
    cs = c[order]
    tp = np.r_[0, np.cumsum(((y == 1) & (d == 1))[order])]
    fp = np.r_[0, np.cumsum(((y == 0) & (d == 1))[order])]
    fn = np.r_[0, np.cumsum(((y == 1) & (d == 0))[order])]
    start = np.searchsorted(cs, cuts, side="left")      # first retained row per cut
    TP, FP, FN = tp[-1] - tp[start], fp[-1] - fp[start], fn[-1] - fn[start]
    n_kept = len(c) - start
    denom = 5 * TP + 4 * FN + FP                          # F2 = 5TP / (5TP + 4FN + FP)
    f2 = np.where(denom > 0, 5 * TP / np.maximum(denom, 1), 0.0)
    return n_kept / len(c), f2


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
    ``max(conf1, conf0) < min_weighted_confidence``). ``grid=None`` searches
    every distinct value of ``confidence`` (exact); an explicit ``grid`` is
    searched with the same rule. Returns {min_confidence, train_f2,
    train_coverage, search, n_candidates, n_eligible, sweep} or None when no
    cut reaches ``min_coverage``. ``sweep`` is a DataFrame, one row per
    candidate cut (ascending).
    """
    y = np.asarray(y_true).astype(int)
    d = np.asarray(pred).astype(int)
    c = np.asarray(confidence, dtype=float)
    if not (len(y) == len(d) == len(c)):
        raise ValueError("y / pred / confidence lengths differ")
    if len(c) == 0:
        return None
    if not np.isfinite(c).all():
        raise ValueError("confidence contains non-finite values")
    if grid is None:
        cuts, search = np.unique(c), ABSTENTION_SEARCH
    else:
        cuts, search = np.unique(np.asarray(grid, dtype=float)), "explicit grid"
    coverage, f2 = _retained_curve(y, d, c, cuts)
    eligible = coverage >= min_coverage
    sweep = pd.DataFrame({"min_confidence": cuts, "coverage": coverage, "f2": f2,
                          "eligible": eligible})
    if not eligible.any():
        return None
    best_f2 = f2[eligible].max()
    j = int(np.flatnonzero(eligible & (f2 == best_f2))[0])   # ties → smaller cut
    return {"min_confidence": float(cuts[j]), "train_f2": float(f2[j]),
            "train_coverage": float(coverage[j]), "search": search,
            "n_candidates": int(len(cuts)), "n_eligible": int(eligible.sum()), "sweep": sweep}
