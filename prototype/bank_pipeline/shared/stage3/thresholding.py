"""
shared.stage3.thresholding
==========================
Operating-point selection: the F-beta grid search (``F2ThresholdSelector``,
built on ``shared.thresholds.sweep_f_beta``), fold-nested OOF thresholds, the
test-oracle diagnostic, and the decision / margin / confidence columns derived
from a threshold (``decision_block``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from shared.thresholds import sweep_f_beta


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

        # The sweep itself is shared.thresholds.sweep_f_beta (Stages 1–2 use
        # it too); Stage 3 passes its own grid. Ties → lowest threshold.
        sweep = sweep_f_beta(y_arr, p, beta=self.beta, grid=self.grid)
        if sweep.empty:
            best_t, best_score = 0.5, 0.0      # same fallback as shared
        else:
            best = sweep.loc[sweep["f_beta"].idxmax()]
            best_t, best_score = float(best["threshold"]), float(best["f_beta"])
        if not keep_sweep:
            sweep = None

        return ThresholdChoice(
            threshold=best_t,
            score=best_score,
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


def decision_block(proba: np.ndarray, threshold) -> dict[str, np.ndarray]:
    """
    Binary decision, signed margin and normalised confidence.

    ``threshold`` is a scalar or per-row array. ``margin = p - t``;
    ``confidence`` rescales |margin| by the room on that side of the boundary
    so it lands in [0, 1] wherever the threshold sits.
    """
    p = np.asarray(proba, dtype=float)
    t = np.broadcast_to(np.asarray(threshold, dtype=float), p.shape)
    decision = (p >= t).astype(int)
    margin = p - t
    upper = np.maximum(1.0 - t, 1e-12)
    lower = np.maximum(t, 1e-12)
    confidence = np.where(margin >= 0, margin / upper, -margin / lower)
    return {
        "decision": decision,
        "margin": margin,
        "confidence": np.clip(confidence, 0.0, 1.0),
        "state": np.where(decision == 1, "flagged", "not_flagged").astype(object),
    }
