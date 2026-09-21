"""
blackbox_pipeline.models.xgb.thresholds
========================================
Operating-point selection.

The rule is the EBM's, bound for bound: maximise F-beta (beta=2) over a linear
grid on ``[0.10, 0.90]`` at 81 candidate values. ``F2ThresholdSelector`` is a
direct translation of ``glass_pipeline.ebm.evaluation.find_optimal_threshold``.

The one deliberate difference is the data the grid is scored against.

    GLASS:  find_optimal_threshold(y_test, test_probs)
    here:   select(y_train, oof_proba)

PR 3 requires that the held-out split take no part in choosing the operating
point. Scoring the grid on ``y_test`` makes every threshold-dependent test
metric — recall, precision, F1, F2, the confusion matrix, the positive
prediction rate — an optimistic number, because the threshold was fitted to the
labels it is then judged on.

For the comparison the stage records, in ``artifact.threshold``:

``"oof"``
    the operating point, selected on ALL out-of-fold training predictions.
    Test-side decisions and the headline test metrics use it.
``"oof_nested"``
    per-fold thresholds (``select_nested``), each chosen without that fold's
    labels. With ``config.nested_oof_threshold=True`` (default) these drive
    the TRAIN-side decision / margin / confidence columns Stage 4 trains on.
``"oof_calibrated_space"``
    the same rule on the calibrated OOF column, for calibrated-space metrics.
``"test_oracle"``
    the value the GLASS rule would have produced on the test split. Its test
    F2 is an upper bound over the grid; reference only, never used to make a
    decision. A wide gap from ``"oof"`` suggests an unstable operating point.

Ties in F-beta resolve to the LOWEST threshold (first ``argmax``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

__all__ = ["ThresholdChoice", "F2ThresholdSelector"]


@dataclass
class ThresholdChoice:
    """A selected threshold plus the sweep that justified it."""

    threshold: float
    score: float
    beta: float
    selected_on: str
    n_rows: int
    grid_low: float
    grid_high: float
    grid_steps: int
    sweep: Optional[pd.DataFrame] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "sweep"}

    def describe(self) -> str:
        return (
            f"threshold {self.threshold:.4f} "
            f"(F{self.beta:g}={self.score:.4f}, selected on {self.selected_on}, "
            f"n={self.n_rows:,})"
        )


class F2ThresholdSelector:
    """
    Grid search for the F-beta-maximising decision threshold.

    Mirrors the EBM's linear scan rather than, say, a PR-curve argmax: the two
    arms must pick their operating point by the same procedure for the
    resulting metrics to be comparable.
    """

    def __init__(
        self,
        beta: float = 2.0,
        low: float = 0.10,
        high: float = 0.90,
        steps: int = 81,
    ):
        if not 0.0 < low < high < 1.0:
            raise ValueError(f"require 0 < low < high < 1, got {low} / {high}")
        if steps < 2:
            raise ValueError("steps must be >= 2")
        self.beta = float(beta)
        self.low = float(low)
        self.high = float(high)
        self.steps = int(steps)

    # ------------------------------------------------------------------
    @property
    def grid(self) -> np.ndarray:
        return np.linspace(self.low, self.high, self.steps)

    def select(
        self,
        y_true,
        proba: np.ndarray,
        selected_on: str = "out-of-fold training predictions",
        keep_sweep: bool = True,
    ) -> ThresholdChoice:
        y_arr = np.asarray(y_true).astype(int)
        p = np.asarray(proba, dtype=float)
        if len(y_arr) != len(p):
            raise ValueError(f"length mismatch: y {len(y_arr)}, proba {len(p)}")
        if len(p) == 0:
            raise ValueError("empty population")
        if np.isnan(p).any():
            raise ValueError(
                "probabilities contain NaN — pass only scored rows."
            )

        grid = self.grid
        scores = np.array([
            fbeta_score(y_arr, (p >= t).astype(int),
                        beta=self.beta, zero_division=0)
            for t in grid
        ])
        best = int(np.argmax(scores))

        sweep = None
        if keep_sweep:
            preds = p[None, :] >= grid[:, None]
            sweep = pd.DataFrame({
                "threshold": grid,
                "fbeta": scores,
                "n_flagged": preds.sum(axis=1),
                "positive_rate": preds.mean(axis=1),
            })

        return ThresholdChoice(
            threshold=float(grid[best]),
            score=float(scores[best]),
            beta=self.beta,
            selected_on=selected_on,
            n_rows=len(p),
            grid_low=self.low,
            grid_high=self.high,
            grid_steps=self.steps,
            sweep=sweep,
        )

    # ------------------------------------------------------------------
    def select_nested(self, y_true, proba: np.ndarray, folds) -> dict:
        """
        Fold-nested thresholds for the training-side decision columns.

        For each fold *k*, the grid is scored on the OTHER folds' OOF rows
        only, and that threshold is applied to fold *k*. No row's label
        influences the threshold used to make its own decision — the same
        pattern as the nested calibration.

        Returns ``{"per_row": ndarray, "by_fold": [...], "mean", "std",
        "min", "max"}``. The spread across folds is a stability diagnostic
        for the operating point.
        """
        y_arr = np.asarray(y_true).astype(int)
        p = np.asarray(proba, dtype=float)
        folds.assert_matches(len(p), "nested threshold selection")
        per_row = np.full(len(p), np.nan, dtype=float)
        by_fold: list[float] = []
        for train_idx, val_idx in folds:
            t = self.select(y_arr[train_idx], p[train_idx],
                            keep_sweep=False).threshold
            per_row[val_idx] = t
            by_fold.append(float(t))
        if np.isnan(per_row).any():
            raise AssertionError(
                "Nested threshold selection left rows unassigned."
            )
        arr = np.asarray(by_fold)
        return {
            "per_row": per_row,
            "by_fold": by_fold,
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    # ------------------------------------------------------------------
    def oracle_on_test(self, y_test, test_proba: np.ndarray) -> ThresholdChoice:
        """
        The value the GLASS rule would produce on the test split.

        Diagnostic only. Nothing downstream may consume it — it is recorded so
        the write-up can quantify how much the GLASS protocol's test-fitted
        threshold flatters its own metrics.
        """
        return self.select(
            y_test, test_proba,
            selected_on="TEST split (oracle — reference only)",
            keep_sweep=False,
        )